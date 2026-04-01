import logging
import math
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import pywt
from librosa import power_to_db
from librosa.feature import melspectrogram
from matplotlib import pyplot as plt

from benchmark.automl_forecasting import convert_tsf_to_dataframe, FREQUENCY_MAP
from benchmark.automl_forecasting import frequencies
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH

eeg_windows = {
    '10s': (4000, 6000),  # Middle 10s
    '30s': (2000, 8000),  # Middle 30s
    '50s': (0, 10000)  # Entire sample (50s)
}

spec_windows = {
    '10m': (-300, 300),  # Entire sample
    '5m': (-150, 150),
    '1m': (-30, 30),
    '10s': (-5, 5),
    '20s': (-10, 10),
    '30s': (-15, 15),
    'pre': (-300, -10),
    'post': (10, 300)

}

eeg_built_spec_windows = {
    '50s': (0, 256),  # Entire sample
    '10s': (100, -100),  # 10s
    'pre': (0, 100),
    'post': (-100, 256)
}

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 4), dtype='float32')

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET is not None:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) //
                256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128)
            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = power_to_db(
                mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

    return img


class ReadData:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def _read_data(self, data_type, file_id):

        if self.is_train:
            PATH = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/train_{data_type}/{file_id}.parquet"
        else:
            PATH = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/test_{data_type}/{file_id}.parquet"

        return pd.read_parquet(PATH)

    def read_spectrogram_data(self, spectrogram_id):
        return self._read_data(
            'spectrograms',
            spectrogram_id).set_index('time')

    def read_eeg_data(self, eeg_id) -> pd.DataFrame:
        return self._read_data('eegs', eeg_id)

    def read_eeg_built_spectrogram_data(self, eeg_id) -> pd.DataFrame:

        montages = ['LL', 'LP', 'RP', 'RR']
        spec = pd.DataFrame()

        if self.is_train:
            _ = PROJECT_PATH + \
                f"/data/hms-harmful-brain-activity-classification/EEG_Spectrograms/{eeg_id}.npy"
            eeg_specs = np.load(_)
        else:
            eeg_specs = spectrogram_from_eeg(
                f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet")

        for i in range(len(montages)):
            spec = pd.concat([spec, pd.DataFrame(
                eeg_specs[:, :, i]).T.add_prefix(f'{montages[i]}_')], axis=1)

        return spec

    def read_train_data(self):
        train_path = PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/train.csv'
        dataframe = pd.read_csv(train_path)
        # TARGETS = ['target', 'fold', 'eeg_id']
        # EEG_IDS = dataframe.eeg_id.unique()
        # target_df = dataframe[TARGETS]
        return dataframe

    def read_test_data(self):
        PROJECT_PATH + '/data/hms-harmful-brain-activity-classification/test.csv'
        return pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv")


class FeatureEngineerData(ReadData):
    """
    Class to engineer features from the EEG and Spectrogram data

    Args:
        metadata (dict): Contains the information on the eeg ids and labels
        is_train (bool): Whether the data is train or test data
        row_id (str): The name of the row id in the metadata

    """

    def __init__(self, metadata, is_train=True, row_id='label_id'):
        super().__init__(is_train)
        self.metadata = metadata
        self.is_train = is_train

        self.row_id = metadata[row_id]

    def get_mean(self, df) -> pd.DataFrame:
        return (df
                .mean()
                .reset_index()
                .set_axis(['var', 'mean'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='mean', index='row_id')
                .add_prefix('mean_')
                )

    def get_max(self, df) -> pd.DataFrame:
        return (df
                .max()
                .reset_index()
                .set_axis(['var', 'max'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='max', index='row_id')
                .add_prefix('max_')
                )

    def get_min(self, df) -> pd.DataFrame:
        return (df
                .max()
                .reset_index()
                .set_axis(['var', 'min'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='min', index='row_id')
                .add_prefix('min_')
                )

    def get_corr(self, df) -> pd.DataFrame:
        """
        Returns the correlation of an eeg file
        """

        def apply_mask(df):
            mask = np.triu(np.ones_like(df, dtype=bool))
            return df.where(mask).unstack().dropna()

        return (df
                .corr()
                .pipe(apply_mask)
                .reset_index()
                .set_axis(['var_1', 'var_2', 'corr'], axis=1)
                .query("var_1 != var_2")
                .assign(
                    row_id=self.row_id,
                    label=lambda x: x.var_1 + "_" + x.var_2
                )
                .pivot(columns='label', values='corr', index='row_id')
                .add_prefix('cor_')
                )

    def filter_spectrogram_corr(self, corr_df) -> pd.DataFrame:
        """
        Returns a dataframe with only the correlation across the same frequency
        """
        return corr_df[[col for col in corr_df.columns if col.split('_')[
            2] == col.split('_')[4]]]

    def filter_eegspectrogram_corr(self, corr_df) -> pd.DataFrame:
        pass

    def get_std(self, df) -> pd.DataFrame:
        return (df
                .std()
                .reset_index()
                .set_axis(['var', 'std'], axis=1)
                .assign(row_id=self.row_id)
                .pivot(columns='var', values='std', index='row_id')
                .add_prefix('std_')
                )

    def get_range(self, df) -> pd.DataFrame:
        return (
            df
            .max()
            .sub(df.min())
            .reset_index()
            .set_axis(['var', 'range'], axis=1)
            .assign(row_id=self.row_id)
            .pivot(columns='var', values='range', index='row_id')
            .add_prefix('range_')
        )


class EEGFeatures(FeatureEngineerData):

    def get_offset(self):
        if self.metadata.get('right_eeg_index') is None:
            return [0, 10000]
        else:
            return [
                self.metadata['left_eeg_index'],
                self.metadata['right_eeg_index']]

    def format_eeg_data(self, window_sizes={}):

        offset_range = self.get_offset()

        df = self.read_eeg_data(
            self.metadata['eeg_id']).iloc[offset_range[0]:offset_range[1]]

        eeg_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            eeg_df = pd.concat([
                eeg_df,
                self.get_features(
                    df.iloc[left_index:right_index], time_id=window)
            ], axis=1)

        return eeg_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_range(df),
                self.get_corr(df)
            ], axis=1).add_prefix(f"eeg_{time_id}_")
        )


class SpectrogramFeatures(FeatureEngineerData):

    def get_offset(self):
        if self.metadata.get('spectrogram_label_offset_seconds') is None:
            return 0
        else:
            return self.metadata['spectrogram_label_offset_seconds']

    def format_spectrogram_data(self, window_sizes={}):

        # Create a variable to make the code more readable
        offset = self.get_offset()

        # Read specific spectrogram window
        df = (self.read_spectrogram_data(self.metadata['spectrogram_id'])
              .loc[offset:offset + 600]
              .fillna(0)
              )

        # Creates the middle of the spectrogram
        middle = (offset + (600 + offset)) / 2

        spec_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            spec_df = pd.concat([
                spec_df,
                self.get_features(
                    df.loc[middle + left_index:middle + right_index], time_id=window)
            ], axis=1)

        return spec_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_min(df),
                self.get_range(df)
            ], axis=1).add_prefix(f"spec_{time_id}_")
        )


class EEGBuiltSpectrogramFeatures(FeatureEngineerData):
    def format_custom_spectrogram(self, window_sizes={()}):
        df = self.read_eeg_built_spectrogram_data(
            self.metadata['eeg_id']).copy()

        spec_df = pd.DataFrame()
        for window in window_sizes:
            left_index = window_sizes[window][0]
            right_index = window_sizes[window][1]

            spec_df = pd.concat([
                spec_df,
                self.get_features(
                    df.iloc[left_index:right_index], time_id=window)
            ], axis=1)

        return spec_df

    def get_features(self, df, time_id) -> pd.DataFrame():
        return (
            pd.concat([
                self.get_mean(df),
                self.get_std(df),
                self.get_max(df),
                self.get_min(df),
                self.get_range(df)
            ], axis=1).add_prefix(f"eegspec_{time_id}_")
        )


class DatasetFormatting:
    """Methods for formatting raw datasets in preparation for modelling."""

    default_start_timestamp = datetime.strptime('1970-01-01 00-00-00', '%Y-%m-%d %H-%M-%S')

    @staticmethod
    def format_univariate_forecasting_data(data_dir, return_df: bool = False):

        headers_and_timestamps = 'libra' not in data_dir  # Libra dataset is missing indices and headers

        meta_data = {
            'file': [],
            'horizon': [],
            'frequency': [],
            'nan_count': [],
            'num_rows': [],
            'num_cols': [],
        }

        if not headers_and_timestamps:  # I-SEM data
            meta_data['origin_index'] = []
            meta_data['step_size'] = []
        try:
            files = os.listdir(data_dir)
        except BaseException:
            files = os.listdir(os.path.join(PROJECT_PATH, data_dir))

        csv_files = [f for f in files if '0_metadata.csv' not in f and f.endswith('csv')]
        df_dict = {}
        for csv_file in csv_files:

            # Read data into a DataFrame
            csv_path = os.path.join(PROJECT_PATH, data_dir, csv_file)
            if headers_and_timestamps:  # I-SEM data
                df = pd.read_csv(csv_path)
                df = df.set_index('applicable_date')
                assert df.shape[1] == 1

                meta_data['file'].append(csv_file)
                meta_data['horizon'].append(24)  # hourly data
                meta_data['frequency'].append(24)  # hourly data
                meta_data['nan_count'].append(int(df.isna().sum()))
                meta_data['num_rows'].append(df.shape[0])
                meta_data['num_cols'].append(df.shape[1])

            else:  # Libra dataset
                df = pd.read_csv(os.path.join(PROJECT_PATH, data_dir, csv_file), header=None)
                assert df.shape[1] == 1

                # The horizon/frequency are based on the paper:
                # "Libra: A Benchmark for Time Series Forecasting Methods" Bauer 2021
                #
                # - "the horizon is 20% of the time series length"
                #
                # - "the [rolling origin] starting point is set either to [a maximum of] 40% of the time series or at two
                #    times the frequency of the time series plus 1"
                #
                # - "the range between the starting point and endpoint is divided into 100 [equal (rounded up)] parts"
                #
                frequency = frequencies[csv_file]
                meta_data['file'].append(csv_file)
                # meta_data['horizon'].append(int(df.shape[0] * 0.2))
                meta_data['horizon'].append(int(min(df.shape[0] * 0.2, 10 * frequency)))
                meta_data['frequency'].append(frequency)
                meta_data['nan_count'].append(int(df.isna().sum()))
                meta_data['num_rows'].append(df.shape[0])
                meta_data['num_cols'].append(df.shape[1])
                meta_data['origin_index'].append(int(max(df.shape[0] * 0.4, (2 * frequency) + 1)))
                meta_data['step_size'].append(math.ceil((0.8 * df.shape[0]) / 100))
            df_dict.update({csv_file: df})
        metadata_df = pd.DataFrame(meta_data)
        metadata_df.to_csv(os.path.join(PROJECT_PATH, data_dir, '0_metadata.csv'), index=False)
        df_dict = None if not return_df else df_dict
        return metadata_df, df_dict

    @staticmethod
    def format_global_forecasting_data(data_dir, gather_metadata=False):
        """Prepare forecasting data for modelling from zip files

        :param str data_dir: Path to data directory
        :param bool gather_metadata: Store datasets metadata in a CSV file, defaults to False
        """

        tsf_files = DatasetFormatting.extract_forecasting_data(data_dir)

        if gather_metadata:
            meta_data = {
                'file': [],
                'frequency': [],
                'horizon': [],
                'has_nans': [],
                'equal_length': [],
                'num_rows': [],
                'num_cols': [],
            }

        # Parse .tsf files sequentially
        for tsf_file in tsf_files:
            csv_path = os.path.join(data_dir, f'{tsf_file.split(".")[0]}.csv')

            # Parse .tsf files and output dataframe
            if not os.path.exists(csv_path) or gather_metadata:
                data, freq, horizon, has_nans, equal_length = convert_tsf_to_dataframe(
                    os.path.join(data_dir, tsf_file), 'NaN', 'value')

                if horizon is None:
                    horizon = DatasetFormatting.select_horizon(freq, csv_path)

                if gather_metadata:
                    meta_data['file'].append(tsf_file)
                    meta_data['horizon'].append(horizon)
                    meta_data['frequency'].append(freq)
                    meta_data['has_nans'].append(has_nans)
                    meta_data['equal_length'].append(equal_length)

                # if not os.path.exists(csv_path):
                # Determine frequency
                if freq is not None:
                    freq = FREQUENCY_MAP[freq]
                else:
                    freq = '1Y'

                # Parse data one variable at time
                df = pd.DataFrame()
                columns = []
                for row_index in range(len(data)):
                    # Convert TSF row to CSV column
                    column = DatasetFormatting.process_row(data, row_index, freq)
                    columns.append(column)

                    if row_index % 1000 == 0:
                        df = pd.concat([df] + columns, axis=1)
                        columns = []
                if len(columns) > 0:
                    df = pd.concat([df] + columns, axis=1)

                df.to_csv(csv_path)
                if gather_metadata:
                    meta_data['num_rows'].append(df.shape[0])
                    meta_data['num_cols'].append(df.shape[1])

        # Save dataset-specific metadata
        if gather_metadata:
            metadata_df = pd.DataFrame(meta_data)
            metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)

    @staticmethod
    def select_horizon(freq, csv_path):
        """Select horizon for forecasters for a given dataset

        :param freq: Time series frequency (str)
        :param csv_path: Path to dataset (str)
        :raises ValueError: If freq is None or not supported
        :return: Forecasting horizon (int)
        """
        if '4_seconds' in csv_path:
            horizon = 15  # i.e. 1 minute
        elif '10_minutes' in csv_path:
            horizon = 6  # i.e. 1 hour

        # The following horizons are suggested by Godahewa et al. (2021)
        elif 'solar_weekly_dataset' in csv_path:
            horizon = 5
        elif freq is None:
            raise ValueError('No frequency or horizon found in file')
        elif freq == 'monthly':
            horizon = 12
        elif freq == 'weekly':
            horizon = 8
        elif freq == 'daily':
            horizon = 30
        elif freq == 'hourly':
            horizon = 168  # i.e. one week
        elif freq == 'half_hourly':
            horizon = 168 * 2  # i.e. one week
        elif freq == 'minutely':
            horizon = 60 * 168  # i.e. one week
        else:
            raise ValueError(f'Unclear what horizon to assign for frequency {freq}')
        return horizon

    @staticmethod
    def process_row(data, row_index, freq):
        """Convert Dataframe row to column with correct timestamp as index

        :param data: Original dataframe
        :param row_index: Index of row to process
        :param freq: Frequency of data
        :raises ValueError: Raised if dates exceed bounds processable by pandas
        :return: Pandas series
        """

        series = data.iloc[row_index, :]
        # Find series name, values and starting timestamp
        series_name = series.loc['series_name']
        values = series.loc['value']
        if 'start_timestamp' in data.columns:
            start_timestamp = series.loc['start_timestamp']
        else:
            start_timestamp = DatasetFormatting.default_start_timestamp

        # Format and apply date range index
        column = pd.DataFrame({series_name: values})
        for i in range(len(column)):
            try:
                # Create a datetime index
                timestamps = pd.date_range(start=start_timestamp, periods=len(column) - i, freq=freq)

                # Truncating if too far into future
                if i > 0:
                    # Truncate by one extra period
                    timestamps = pd.date_range(start=start_timestamp, periods=len(column) - (i + 1), freq=freq)
                    logging.warning(f'Truncating {series_name} from {len(column)} to {len(column) - (i + 1)}')
                    column = column.head(len(column) - (i + 1))

                column = column.set_index(timestamps)
                break
            except pd.errors.OutOfBoundsDatetime as e:
                if i == len(column) - 1:
                    logging.error(series_name, start_timestamp, len(column), freq)
                    raise ValueError('Dates too far into the future for pandas to process') from e

        # Aggregate rows where timestamp index is apart by 1 second in hourly data
        if freq == '1H':
            column = column.resample('1H').mean()

        return column

    @staticmethod
    def extract_forecasting_data(data_dir):
        """Read zip files from directory and extract .tsf files

        :param data_dir: Path to data directory of zip files
        :raises NotADirectoryError: occurs if non-directory passed as parameter
        :raises IOError: occurs if no zip files found
        :return: list of paths (str) to extracted .tsf files
        """
        # Validate input directory path
        try:
            zip_files = [f for f in os.listdir(data_dir) if f.endswith('zip')]
        except NotADirectoryError as e:
            raise NotADirectoryError('\nProvide a path to a directory of zip files (of forecasting data)') from e

        if len(zip_files) == 0:
            raise IOError(f'\nNo zip files found in "{data_dir}"')

        # Extract zip files
        for filename in zip_files:
            with zipfile.ZipFile(os.path.join(data_dir, filename)) as zip_file:
                files = zip_file.namelist()
                error_msg = 'Zip files expected to contain exactly one .tsf file'
                assert len(files) == 1 and files[0].endswith('tsf'), error_msg
                output_file = os.path.join(data_dir, files[0])
                if not os.path.exists(output_file):
                    zip_file.extractall(data_dir)

        tsf_files = [f for f in os.listdir(data_dir) if f.endswith('tsf')]
        return tsf_files

    @staticmethod
    def format_anomaly_data(data_dir):
        """Format anomaly detection datasets

        :param data_dir: Path to directory of datasets
        """

        DatasetFormatting.format_3W_data(data_dir)
        DatasetFormatting.format_falling_data(data_dir)
        DatasetFormatting.format_BETH_data(data_dir)
        DatasetFormatting.format_HAI_data(data_dir)
        DatasetFormatting.format_NAB_data(data_dir)
        DatasetFormatting.format_SKAB_data(data_dir)

    @staticmethod
    def format_3W_data(data_dir):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        """
        os.path.join(data_dir, '3W')

    @staticmethod
    def format_falling_data(data_dir):
        """Format falling data

        :param data_dir: Path to directory of datasets
        """

    @staticmethod
    def format_BETH_data(data_dir):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        """

    @staticmethod
    def format_HAI_data(data_dir):
        """Format HAI Security Dataset data

        :param data_dir: Path to directory of datasets
        """

    @staticmethod
    def format_NAB_data(data_dir):
        """Format Numenta Anomaly detection Benchmark data

        :param data_dir: Path to directory of datasets
        """

    @staticmethod
    def format_SKAB_data(data_dir):
        """Format Skoltech Anomaly Benchmark data

        :param data_dir: Path to directory of datasets
        """
