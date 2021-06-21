from examples.sensitivity_analysis.dataset_access import get_scoring_data
from examples.sensitivity_analysis.chains_access import get_three_depth_manual_class_chain
from fedot.sensitivity.deletion_methods.multi_times_analysis import MultiTimesAnalyze
from fedot.core.data.data_split import train_test_data_setup


def run_mta_analysis(chain, train_data, test_data, valid_data):
    size_reduction_ratio = MultiTimesAnalyze(chain=chain, train_data=train_data, test_data=test_data,
                                             valid_data=valid_data, case_name='scoring_mta_experiment').\
        analyze(is_visualize=True)
    print(f'The number of deleted nodes to the original chain length is {size_reduction_ratio}')


if __name__ == '__main__':
    chain = get_three_depth_manual_class_chain()
    train_data, test_data = get_scoring_data()
    test_data, valid_data = train_test_data_setup(test_data, split_ratio=0.5)

    run_mta_analysis(chain, train_data, test_data, valid_data)
