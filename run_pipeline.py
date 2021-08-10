import configparser
import sys

from fedot.core.data.data import InputData

from fedot.core.data.data_split import train_test_data_setup

from fedot.core.pipelines.pipeline import Pipeline


def extract_data_from_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    pipeline_file = config['DEFAULT']['pipeline_file_path']
    input_data = config['DEFAULT']['data']
    output_path = config['DEFAULT']['output_path']

    return pipeline_file, input_data, output_path


def run_fedot(config_file):
    pipeline_file, input_data_file, output_path = extract_data_from_config_file(config_file)

    pipeline = Pipeline().load(pipeline_file)

    input_data = InputData.from_csv(input_data_file)

    train_data, test_data = train_test_data_setup(input_data)

    pipeline.fit_from_scratch(train_data)

    pipeline.save(output_path)


if __name__ == '__main__':
    config_file = sys.argv[1]
    run_fedot(config_file)
