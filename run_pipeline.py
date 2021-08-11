import configparser
import sys

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline


def extract_data_from_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    pipeline_file = config['DEFAULT']['pipeline_file_path']
    input_data = config['DEFAULT']['data']
    output_path = config['DEFAULT']['output_path']

    if 'OPTIONAL' in config.sections():
        if config['OPTIONAL']['test_data']:
            test_data_path = config['OPTIONAL']['test_data']
            return pipeline_file, input_data, test_data_path, output_path

    return pipeline_file, input_data, None, output_path


def run_fedot(config_file):
    pipeline_file, train_data_path, test_data_path, output_path = extract_data_from_config_file(config_file)

    pipeline = Pipeline().load(pipeline_file)

    train_data = InputData.from_csv(train_data_path)

    pipeline.fit_from_scratch(train_data)

    if test_data_path:
        test_data = InputData.from_csv(test_data_path)
        pipeline.predict(test_data)

    pipeline.save(output_path)


if __name__ == '__main__':
    config_file = sys.argv[1]
    run_fedot(config_file)
