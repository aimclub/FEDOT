import configparser
import sys

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline


def extract_data_from_config_file(file):
    config = configparser.ConfigParser()
    config.read(file, encoding='utf-8')
    pipeline_file_path = config['DEFAULT']['pipeline_file_path']
    input_data = config['DEFAULT']['train_data']
    task = eval(config['DEFAULT']['task'])
    output_path = config['DEFAULT']['output_path']

    test_data_path = config['OPTIONAL'].get('test_data')

    return pipeline_file_path, input_data, task, test_data_path, output_path


def run_fedot(config_file):
    pipeline_file_path, train_data_path, \
    task, test_data_path, output_path = extract_data_from_config_file(config_file)

    pipeline = Pipeline()
    pipeline.load(pipeline_file_path)

    train_data = InputData.from_csv(file_path=train_data_path,
                                    task=task)

    pipeline.fit_from_scratch(train_data)

    if test_data_path:
        test_data = InputData.from_csv(test_data_path)
        pipeline.predict(test_data)

    pipeline.save(path=output_path)


if __name__ == '__main__':
    config_file = sys.argv[1]
    run_fedot(config_file)
