from examples.advanced.sensitivity_analysis.dataset_access import get_scoring_data
from examples.advanced.sensitivity_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.data.data_split import train_test_data_setup
from fedot.sensitivity.deletion_methods.multi_times_analysis import MultiTimesAnalyze


def run_mta_analysis(pipeline, train_data, test_data, valid_data):
    size_reduction_ratio = MultiTimesAnalyze(pipeline=pipeline, train_data=train_data, test_data=test_data,
                                             valid_data=valid_data, case_name='scoring_mta_experiment'). \
        analyze(visualization=True)
    print(f'The number of deleted nodes to the original pipeline length is {size_reduction_ratio}')


if __name__ == '__main__':
    pipeline = get_three_depth_manual_class_pipeline()
    train_data, test_data = get_scoring_data()
    test_data, valid_data = train_test_data_setup(test_data, split_ratio=0.5)

    run_mta_analysis(pipeline, train_data, test_data, valid_data)
