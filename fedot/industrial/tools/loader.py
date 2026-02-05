import logging
import os
import shutil
import urllib.request as request
import zipfile
from pathlib import Path
from typing import Optional, Union

import chardet
import pandas as pd
from datasets import load_dataset
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4
from datasetsforecast.m5 import M5
from scipy.io.arff import loadarff
from sktime.datasets import load_from_tsfile_to_dataframe
from tqdm import tqdm

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import M4_PREFIX
from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH, EXAMPLES_DATA_PATH


class DataLoader:
    """Class for reading data files and downloading from UCR archive if not found locally.
    At the moment supports ``.ts``, ``.txt``, ``.tsv``, and ``.arff`` formats.

    Args:
        dataset_name: name of dataset
        folder: path to folder with data

    Examples:
        >>> data_loader = DataLoader('ItalyPowerDemand')
        >>> train_data, test_data = data_loader.load_data()
    """

    def __init__(self, dataset_name: str, folder: Optional[str] = None, source_url: Optional[str] = None):
        self.logger = logging.getLogger('DataLoader')
        self.url = source_url if source_url is not None else f'http://www.timeseriesclassification.com/aeon-toolkit/'
        self.dataset_name = dataset_name
        self.folder = folder
        self.forecast_data_source = {
            'M3': M3.load,
            'M4': M4.load,
            # 'M4': self.local_m4_load,
            'M5': M5.load,
            'monash_tsf': load_dataset
        }

    def load_forecast_data(self, forecast_family: Optional[str] = None, folder: Optional[Union[Path, str]] = None):
        if forecast_family not in self.forecast_data_source:
            forecast_family = self.dataset_name.get('benchmark') if isinstance(self.dataset_name, dict) else 'M4'
        if folder is None:
            folder = EXAMPLES_DATA_PATH
        loader = self.forecast_data_source[forecast_family]
        dataset_name = self.dataset_name.get('dataset') if isinstance(self.dataset_name, dict) else self.dataset_name
        group_df, _, _ = loader(directory=folder, group=f'{M4_PREFIX[dataset_name[0]]}')
        ts_df = group_df[group_df['unique_id'] == dataset_name]
        del ts_df['unique_id']
        ts_df = ts_df.set_index('datetime') if 'datetime' in ts_df.columns else ts_df.set_index('ds')
        train_data = ts_df.values.flatten()
        target = train_data[-self.dataset_name['task_params']['forecast_length']:].flatten()
        train_data = (train_data, target)
        return train_data, train_data

    @staticmethod
    def local_m4_load(group: Optional[str] = None):
        path_to_result = EXAMPLES_DATA_PATH + '/forecasting/'
        for result_cvs in os.listdir(path_to_result):
            if result_cvs.__contains__(group):
                return pd.read_csv(Path(path_to_result, result_cvs))

    def load_detection_data(self, dataset_name: Optional[dict] = {}):
        if dataset_name is None:
            dataset_name = {}
        folder = dataset_name.get('benchmark', 'valve1')
        dataset = dataset_name.get('dataset', '1')
        path_to_skab_data = EXAMPLES_DATA_PATH + f'/benchmark/detection/data/{folder}/{dataset}.csv'
        df = pd.read_csv(path_to_skab_data, index_col='datetime', sep=';', parse_dates=True)
        train_idx = dataset_name.get('train_data_size', 'anomaly-free')
        if isinstance(train_idx, str):
            train_data = EXAMPLES_DATA_PATH + f'/benchmark/detection/data/{train_idx}/{train_idx}.csv'
            train_data = pd.read_csv(train_data, index_col='datetime', sep=';', parse_dates=True)
            label = np.array([0 for _ in range(len(train_data))])
            return (train_data.values, label), (df.iloc[:, :-2].values, df.iloc[:, -2].values)
        return None, None

    def _load_benchmark_data(self, specific_strategy: str):
        train_data, test_data = None, None
        if specific_strategy == 'anomaly_detection':
            train_data, test_data = self.load_detection_data(self.dataset_name)
        elif specific_strategy in ['ts_forecasting', 'forecasting_assumptions']:
            train_data, test_data = self.load_forecast_data(self.folder)
        elif specific_strategy is not None:
            train_data, test_data = self.load_data(self.dataset_name)
        return train_data, test_data

    def load_custom_data(self, specific_strategy: Optional[str] = None):
        dict_dataset = isinstance(self.dataset_name, dict)
        if dict_dataset and 'train_data' in self.dataset_name.keys():
            return self.dataset_name['train_data'], self.dataset_name['test_data']
        return self._load_benchmark_data(specific_strategy)

    def load_data(self, shuffle: bool = True) -> tuple:
        """Load data for classification experiment locally or externally from UCR archive.

        Returns:
            tuple: train and test data
        """
        dataset_name = self.dataset_name
        data_path = os.path.join(PROJECT_PATH, 'fedot', 'industrial' 'data') if self.folder is None else self.folder
        _, train_data, test_data = self.read_train_test_files(dataset_name=dataset_name,
                                                              data_path=data_path,
                                                              shuffle=shuffle)
        if train_data is None:
            self.logger.info(f'Downloading {dataset_name} from {self.url}...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            for _ in (download_path, temp_data_path):
                os.makedirs(_, exist_ok=True)

            url = self.url + f'/{dataset_name}.zip'
            request.urlretrieve(url, download_path + f'temp_data_{dataset_name}')
            try:
                zipfile.ZipFile(download_path + f'temp_data_{dataset_name}').extractall(temp_data_path + dataset_name)
            except zipfile.BadZipFile:
                raise FileNotFoundError(f'Cannot extract data: {dataset_name} dataset not found in {self.url}')
            else:
                self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
                train_data, test_data = self.extract_data(dataset_name, temp_data_path)
                shutil.rmtree(cache_path)

        self.logger.info('Data read successfully from local folder')

        if isinstance(train_data[0].iloc[0, 0], pd.Series):
            def convert(arr):
                """Transform pd.Series values to np.ndarray"""
                return np.array([d.values for d in arr])
            train_data = (np.apply_along_axis(convert, 1, train_data[0]), train_data[1])
            test_data = (np.apply_along_axis(convert, 1, test_data[0]), test_data[1])

        return train_data, test_data

    def read_train_test_files(self, data_path: Union[Path, str], dataset_name: str, shuffle: bool = True):

        dataset_dir_path = os.path.join(data_path, dataset_name)
        file_path = dataset_dir_path + f'/{dataset_name}_TRAIN'
        is_multivariate = False
        self.logger.info(f'Reading data from {dataset_dir_path}')

        if os.path.isfile(file_path + '.tsv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='tsv')
        elif os.path.isfile(file_path + '.txt'):
            x_train, y_train, x_test, y_test = self.read_txt_files(dataset_name, data_path)
        elif os.path.isfile(file_path + '.ts'):
            x_train, y_train, x_test, y_test = self.read_ts_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.arff'):
            x_train, y_train, x_test, y_test = self.read_arff_files(dataset_name, data_path)
            is_multivariate = True
        elif os.path.isfile(file_path + '.csv'):
            x_train, y_train, x_test, y_test = self.read_tsv_or_csv(dataset_name, data_path, mode='csv')
        else:
            self.logger.error(f'Data not found in {dataset_dir_path}')
            return None, None, None

        y_train, y_test = convert_type(y_train, y_test)

        if shuffle:
            shuffled_idx = np.arange(x_train.shape[0])
            np.random.shuffle(shuffled_idx)
            if isinstance(x_train, pd.DataFrame):
                x_train = x_train.iloc[shuffled_idx, :]
            else:
                x_train = x_train[shuffled_idx, :]
            y_train = y_train[shuffled_idx]
        return is_multivariate, (x_train, y_train), (x_test, y_test)

    @staticmethod
    def predict_encoding(file_path: Union[Path, str], n_lines: int = 20) -> str:
        with Path(file_path).open('rb') as f:
            rawdata = b''.join([f.readline() for _ in range(n_lines)])
        return chardet.detect(rawdata)['encoding']

    def _load_from_tsfile_to_dataframe(
            self,
            full_file_path_and_name,
            return_separate_X_and_y=True,
            replace_missing_vals_with='NaN'):
        """Loads data from a .ts file into a Pandas DataFrame.
        Taken from https://github.com/ChangWeiTan/TS-Extrinsic-Regression/blob/master/utils/data_loader.py

        Args:
            full_file_path_and_name: The full pathname of the .ts file to read. return_separate_X_and_y: true if X
                                     and Y values should be returned as separate Data Frames (X) and a numpy array (y),
                                     false otherwise.
            replace_missing_vals_with: The value that missing values in the text file should be replaced with prior to
                                       parsing.

        Returns:
            If ``return_separate_X_and_y`` then a tuple containing a DataFrame and a numpy array containing the
            relevant time-series and corresponding class values. If not ``return_separate_X_and_y`` then a single
            DataFrame containing all time-series and (if relevant) a column ``class_vals`` the associated class values.

        """

        # Initialize flags and variables used when parsing the file
        metadata_started = False
        data_started = False

        has_problem_name_tag = False
        has_timestamps_tag = False
        has_univariate_tag = False
        has_class_labels_tag = False
        has_target_labels_tag = False
        has_data_tag = False

        previous_timestamp_was_float = None
        previous_timestamp_was_int = None
        previous_timestamp_was_timestamp = None
        num_dimensions = None
        is_first_case = True
        instance_list = []
        class_val_list = []
        line_num = 0
        TsFileParseException = Exception

        encoding = self.predict_encoding(full_file_path_and_name)

        with open(full_file_path_and_name, 'r', encoding=encoding) as file:
            dataset_name = os.path.basename(full_file_path_and_name)
            for line in tqdm(
                    file.readlines(),
                    desc='Loading data',
                    leave=False,
                    postfix=dataset_name,
                    unit='lines'):
                # print(".", end='')
                # Strip white space from start/end of line and change to
                # lowercase for use below
                line = line.strip().lower()
                # Empty lines are valid at any point in a file
                if line:
                    # Check if this line contains metadata
                    # Please note that even though metadata is stored in this function it is not currently
                    # published externally
                    if line.startswith("@problemname"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException(
                                "metadata must come before data")
                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException(
                                "problemname tag requires an associated value")

                        # problem_name = line[len("@problemname") + 1:]
                        has_problem_name_tag = True
                        metadata_started = True
                    elif line.startswith("@timestamps"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException(
                                "metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len != 2:
                            raise TsFileParseException(
                                "timestamps tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            timestamps = True
                        elif tokens[1] == "false":
                            timestamps = False
                        else:
                            raise TsFileParseException(
                                "invalid timestamps value")
                        has_timestamps_tag = True
                        metadata_started = True
                    elif line.startswith("@univariate"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException(
                                "metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)
                        if token_len != 2:
                            raise TsFileParseException(
                                "univariate tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            pass
                        elif tokens[1] == "false":
                            pass
                        else:
                            raise TsFileParseException(
                                "invalid univariate value")

                        has_univariate_tag = True
                        metadata_started = True
                    elif line.startswith("@classlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException(
                                "metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException(
                                "classlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            class_labels = True
                        elif tokens[1] == "false":
                            class_labels = False
                        else:
                            raise TsFileParseException(
                                "invalid classLabel value")

                        # Check if we have any associated class values
                        if token_len == 2 and class_labels:
                            raise TsFileParseException(
                                "if the classlabel tag is true then class values must be supplied")

                        has_class_labels_tag = True
                        class_label_list = [token.strip()
                                            for token in tokens[2:]]
                        metadata_started = True
                    elif line.startswith("@targetlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException(
                                "metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException(
                                "targetlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            target_labels = True
                        elif tokens[1] == "false":
                            target_labels = False
                        else:
                            raise TsFileParseException(
                                "invalid targetLabel value")

                        has_target_labels_tag = True
                        class_val_list = []
                        metadata_started = True
                    # Check if this line contains the start of data
                    elif line.startswith("@data"):
                        if line != "@data":
                            raise TsFileParseException(
                                "data tag should not have an associated value")

                        if data_started and not metadata_started:
                            raise TsFileParseException(
                                "metadata must come before data")
                        else:
                            has_data_tag = True
                            data_started = True
                    # If the 'data tag has been found then metadata has been
                    # parsed and data can be loaded
                    elif data_started:
                        # Check that a full set of metadata has been provided
                        incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or \
                            not has_univariate_tag or not has_target_labels_tag or \
                            not has_data_tag
                        incomplete_classification_meta_data = \
                            not has_problem_name_tag or not has_timestamps_tag \
                            or not has_univariate_tag or not has_class_labels_tag \
                            or not has_data_tag
                        if incomplete_regression_meta_data and incomplete_classification_meta_data:
                            raise TsFileParseException(
                                "a full set of metadata has not been provided before the data")

                        # Replace any missing values with the value specified
                        line = line.replace("?", replace_missing_vals_with)

                        # Check if we dealing with data that has timestamps
                        if timestamps:
                            # We're dealing with timestamps so cannot just split line on ':'
                            # as timestamps may contain one
                            has_another_value = False
                            has_another_dimension = False

                            timestamps_for_dimension = []
                            values_for_dimension = []

                            this_line_num_dimensions = 0
                            line_len = len(line)
                            char_num = 0

                            while char_num < line_len:
                                # Move through any spaces
                                while char_num < line_len and str.isspace(
                                        line[char_num]):
                                    char_num += 1

                                # See if there is any more data to read in or
                                # if we should validate that read thus far

                                if char_num < line_len:

                                    # See if we have an empty dimension (i.e.
                                    # no values)
                                    if line[char_num] == ":":
                                        if len(instance_list) < (
                                                this_line_num_dimensions + 1):
                                            instance_list.append([])

                                        instance_list[this_line_num_dimensions].append(
                                            pd.Series())
                                        this_line_num_dimensions += 1

                                        has_another_value = False
                                        has_another_dimension = True

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                        char_num += 1
                                    else:
                                        # Check if we have reached a class
                                        # label
                                        if line[char_num] != "(" and target_labels:
                                            class_val = line[char_num:].strip()

                                            # if class_val not in class_val_list:
                                            #     raise TsFileParseException(
                                            #         "the class value '" + class_val + "' on line " + str(
                                            # line_num + 1) + " is not valid")

                                            class_val_list.append(
                                                float(class_val))
                                            char_num = line_len

                                            has_another_value = False
                                            has_another_dimension = False

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                                        else:

                                            # Read in the data contained within
                                            # the next tuple

                                            if line[char_num] != "(" and not target_labels:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not start with a '('")

                                            char_num += 1
                                            tuple_data = ""

                                            while char_num < line_len and line[char_num] != ")":
                                                tuple_data += line[char_num]
                                                char_num += 1

                                            if char_num >= line_len or line[char_num] != ")":
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not end with a ')'")

                                            # Read in any spaces immediately
                                            # after the current tuple

                                            char_num += 1

                                            while char_num < line_len and str.isspace(
                                                    line[char_num]):
                                                char_num += 1

                                            # Check if there is another value
                                            # or dimension to process after
                                            # this tuple

                                            if char_num >= line_len:
                                                has_another_value = False
                                                has_another_dimension = False

                                            elif line[char_num] == ",":
                                                has_another_value = True
                                                has_another_dimension = False

                                            elif line[char_num] == ":":
                                                has_another_value = False
                                                has_another_dimension = True

                                            char_num += 1

                                            # Get the numeric value for the tuple by reading from the end
                                            # of the tuple data backwards to
                                            # the last comma

                                            last_comma_index = tuple_data.rfind(
                                                ',')

                                            if last_comma_index == -1:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1)
                                                    + " contains a tuple that has no comma inside of it")

                                            try:
                                                value = tuple_data[last_comma_index + 1:]
                                                value = float(value)

                                            except ValueError:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1)
                                                    + " contains a tuple that does not have a valid numeric value")

                                            # Check the type of timestamp that
                                            # we have

                                            timestamp = tuple_data[0: last_comma_index]

                                            try:
                                                timestamp = int(timestamp)
                                                timestamp_is_int = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_int = False

                                            if not timestamp_is_int:
                                                try:
                                                    timestamp = float(
                                                        timestamp)
                                                    timestamp_is_float = True
                                                    timestamp_is_timestamp = False
                                                except ValueError:
                                                    timestamp_is_float = False

                                            if not timestamp_is_int and not timestamp_is_float:
                                                try:
                                                    timestamp = timestamp.strip()
                                                    timestamp_is_timestamp = True
                                                except ValueError:
                                                    timestamp_is_timestamp = False

                                            # Make sure that the timestamps in the file
                                            # (not just this dimension or case) are consistent

                                            if not timestamp_is_timestamp and not timestamp_is_int \
                                                    and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) +
                                                    " contains a tuple that has an invalid timestamp '"
                                                    + timestamp + "'")

                                            if previous_timestamp_was_float is not None \
                                                    and previous_timestamp_was_float and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) +
                                                    " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_int is not \
                                                    None and previous_timestamp_was_int and not timestamp_is_int:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) +
                                                    " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_timestamp is not None \
                                                    and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) +
                                                    " contains tuples where the timestamp format is inconsistent")

                                            # Store the values

                                            timestamps_for_dimension += [
                                                timestamp]
                                            values_for_dimension += [value]

                                            # If this was our first tuple then
                                            # we store the type of timestamp we
                                            # had

                                            if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                                previous_timestamp_was_timestamp = True
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_int is None and timestamp_is_int:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = True
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_float is None and timestamp_is_float:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = True

                                            # See if we should add the data for
                                            # this dimension

                                            if not has_another_value:
                                                if len(instance_list) < (
                                                        this_line_num_dimensions + 1):
                                                    instance_list.append([])

                                                if timestamp_is_timestamp:
                                                    timestamps_for_dimension = pd.DatetimeIndex(
                                                        timestamps_for_dimension)

                                                instance_list[this_line_num_dimensions].append(
                                                    pd.Series(index=timestamps_for_dimension,
                                                              data=values_for_dimension))
                                                this_line_num_dimensions += 1

                                                timestamps_for_dimension = []
                                                values_for_dimension = []

                                elif has_another_value:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ',' that is not followed by another tuple")

                                elif has_another_dimension and target_labels:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ':' while it should list a class value")

                                elif has_another_dimension and not target_labels:
                                    if len(instance_list) < (
                                            this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(
                                        pd.Series(dtype=np.float32))
                                    this_line_num_dimensions += 1
                                    num_dimensions = this_line_num_dimensions

                                # If this is the 1st line of data we have seen
                                # then note the dimensions

                                if not has_another_value and not has_another_dimension:
                                    if num_dimensions is None:
                                        num_dimensions = this_line_num_dimensions

                                    if num_dimensions != this_line_num_dimensions:
                                        raise TsFileParseException(
                                            "line " +
                                            str(
                                                line_num +
                                                1) +
                                            " does not have the same number of dimensions as the previous line of data")

                            # Check that we are not expecting some more data,
                            # and if not, store that processed above

                            if has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (
                                        this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(
                                    pd.Series())
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then
                            # note the dimensions

                            if not has_another_value and num_dimensions != this_line_num_dimensions:
                                raise TsFileParseException(
                                    "line " +
                                    str(
                                        line_num +
                                        1) +
                                    "does not have the same number of dimensions as the "
                                    "previous line of data")

                            # Check if we should have class values, and if so that they are contained
                            # in those listed in the metadata

                            if target_labels and len(class_val_list) == 0:
                                raise TsFileParseException(
                                    "the cases have no associated class values")
                        else:
                            dimensions = line.split(":")
                            # If first row then note the number of dimensions
                            # (that must be the same for all cases)
                            if is_first_case:
                                num_dimensions = len(dimensions)

                                if target_labels:
                                    num_dimensions -= 1

                                for dim in range(0, num_dimensions):
                                    instance_list.append([])
                                is_first_case = False

                            # See how many dimensions that the case whose data
                            # in represented in this line has
                            this_line_num_dimensions = len(dimensions)

                            if target_labels:
                                this_line_num_dimensions -= 1

                            # All dimensions should be included for all series,
                            # even if they are empty
                            if this_line_num_dimensions != num_dimensions:
                                print(
                                    "inconsistent number of dimensions. Expecting " +
                                    str(num_dimensions) +
                                    " but have read " +
                                    str(this_line_num_dimensions))

                            # Process the data for each dimension
                            for dim in range(0, num_dimensions):
                                try:
                                    dimension = dimensions[dim].strip()

                                    if dimension:
                                        data_series = dimension.split(",")
                                        data_series = [float(i)
                                                       for i in data_series]
                                        instance_list[dim].append(
                                            pd.Series(data_series))
                                    else:
                                        instance_list[dim].append(pd.Series())
                                except Exception:
                                    _ = 1

                            if target_labels:
                                try:
                                    class_val_list.append(
                                        float(dimensions[num_dimensions].strip()))
                                except Exception:
                                    _ = 1

                line_num += 1

        # Check that the file was not empty
        if line_num:
            # Check that the file contained both metadata and data
            complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag \
                and has_target_labels_tag and has_data_tag
            complete_classification_meta_data = \
                has_problem_name_tag and has_timestamps_tag \
                and has_univariate_tag and has_class_labels_tag and has_data_tag

            if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
                raise TsFileParseException("metadata incomplete")
            elif metadata_started and not data_started:
                raise TsFileParseException(
                    "file contained metadata but no data")
            elif metadata_started and data_started and len(instance_list) == 0:
                raise TsFileParseException(
                    "file contained metadata but no data")

            # Create a DataFrame from the data parsed above
            data = pd.DataFrame(dtype=np.float32)

            for dim in range(0, num_dimensions):
                data['dim_' + str(dim)] = instance_list[dim]

            # Check if we should return any associated class labels separately

            if target_labels:
                if return_separate_X_and_y:
                    return data, np.asarray(class_val_list)
                else:
                    data['class_vals'] = pd.Series(class_val_list)
                    return data
            else:
                return data
        else:
            raise TsFileParseException("empty file")

    @staticmethod
    def read_tsv_or_csv(dataset_name: str, data_path: str, mode: str = 'tsv') -> tuple:
        """Read ``tsv`` or ``csv`` file that contains data for classification experiment.
        Data must be placed in ``data`` folder with ``.tsv``/``csv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data
            mode: ``tsv`` or ``csv`` file format
        Returns:
            tuple: (x_train, x_test) and (y_train, y_test)
        """
        def load_process_data(path_to_dataset, sep):
            data = pd.read_csv(path_to_dataset, sep=sep, header=None)
            features = data.iloc[:, 1:]
            target = data[0].values
            try:
                target = target.astype(int)
            except ValueError:
                target = target.astype(str)
            return features, target

        dataset_dir = os.path.join(data_path, dataset_name)
        if mode not in ['tsv', 'csv']:
            raise ValueError(f'Invalid mode {mode}. Should be one of "tsv" or "csv"')
        separator = '\t' if mode == 'tsv' else ','
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.{mode}', separator)
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.{mode}', separator)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_txt_files(dataset_name: str, data_path: str):
        """
        Reads data from ``.txt`` file.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            train and test data tuple
        """
        dataset_dir = os.path.join(data_path, dataset_name)
        data_train = np.genfromtxt(dataset_dir + f'/{dataset_name}_TRAIN.txt')
        data_test = np.genfromtxt(dataset_dir + f'/{dataset_name}_TEST.txt')
        x_train, y_train = data_train[:, 1:], data_train[:, 0]
        x_test, y_test = data_test[:, 1:], data_test[:, 0]
        return x_train, y_train, x_test, y_test

    def read_ts_files(self, dataset_name: str, data_path: str):
        """
        Reads multivariate data from ``.ts`` file
        """
        def load_process_data(path_to_dataset):
            try:
                features, target = load_from_tsfile_to_dataframe(path_to_dataset,
                                                                 return_separate_X_and_y=True)
            except Exception as e:
                self.logger.info(f'Performing custom ts files reading due to {e}')
                features, target = self._load_from_tsfile_to_dataframe(path_to_dataset,
                                                                       return_separate_X_and_y=True)
            return features, target

        dataset_dir = os.path.join(data_path, dataset_name)
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.ts')
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.ts')

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_arff_files(dataset_name, data_path) -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
        """
        Reads multivariate data from ``.arff`` file

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data

        Returns:
            x_train: train dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_train: train target array of shape (n_samples,)
            x_test: test dataframe of shape (n_samples, dim) with pd.Series of shape (ts_length,)
            y_test: test target array of shape (n_samples,)

        """
        def load_process_data(path_to_dataset):
            data, meta = loadarff(path_to_dataset)
            data_array = np.asarray([data[name] for name in meta.names()])
            features, target = data_array[:-1].T.ravel(), data_array[-1]
            is_multivariate = len(features[0].shape)
            if is_multivariate:
                void_free = pd.Series(features).apply(lambda elem: elem.view(np.float64).reshape(elem.shape[0], -1))
                features = pd.DataFrame([[pd.Series(arr[i]) for i in range(arr.shape[0])] for arr in void_free.values])
                return features, target
            return features.astype('float64'), target

        dataset_dir = os.path.join(data_path, dataset_name)
        x_train, y_train = load_process_data(dataset_dir + f'/{dataset_name}_TRAIN.arff')
        x_test, y_test = load_process_data(dataset_dir + f'/{dataset_name}_TEST.arff')

        return x_train, y_train, x_test, y_test

    def extract_data(self, dataset_name: str, data_path: str):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            is_multi, (x_train, y_train), (x_test, y_test) = self.read_train_test_files(
                data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        # Conversion of target values to int or str
        y_train, y_test = convert_type(y_train, y_test)

        # Save data to tsv files
        new_path = os.path.join(PROJECT_PATH, 'fedot', 'industrial', 'data') if self.folder is None else self.folder
        new_path = os.path.join(new_path, dataset_name)
        os.makedirs(new_path, exist_ok=True)

        self.logger.info(f'Saving {dataset_name} data files to {new_path}')
        for subset in ('TRAIN', 'TEST'):
            if not is_multi:
                df = pd.DataFrame(x_train if subset == 'TRAIN' else x_test)
                df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
                df.to_csv(
                    os.path.join(
                        new_path,
                        f'{dataset_name}_{subset}.tsv'),
                    sep='\t',
                    index=False,
                    header=False)
                del df

            else:
                old_path = os.path.join(
                    data_path, dataset_name, f'{dataset_name}_{subset}.ts')
                shutil.move(old_path, new_path)

        if is_multi:
            return (x_train, y_train), (x_test, y_test)
        else:
            return (pd.DataFrame(x_train),
                    y_train), (pd.DataFrame(x_test), y_test)


def convert_type(y_train, y_test):
    # Conversion of target values to int or str
    try:
        y_train = y_train.astype('float')
        y_test = y_test.astype('float')
    except ValueError:
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
    return y_train, y_test
