from fedot import Fedot
from fedot.core.utils import fedot_project_root, set_random_seed


def run_sampling_stage_example(timeout: float = 1.0):
    train_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_test.csv'

    model = Fedot(
        problem='classification',
        timeout=timeout,
        preset='fast_train',
        max_depth=2,
        max_arity=2,
        sampling_config={
            'provider': 'sampling_zoo',
            'strategy': 'random',
            'candidate_ratios': [0.2, 0.3, 0.5],
            'delta_metric_threshold': 0.05,
        }
    )

    model.fit(features=train_data_path, target='target')
    _ = model.predict(features=test_data_path)

    print('Sampling metadata:', model.sampling_stage_metadata)
    print('Metrics:', model.get_metrics())


if __name__ == '__main__':
    set_random_seed(42)

    try:
        run_sampling_stage_example(timeout=1.0)
    except ModuleNotFoundError as ex:
        print('Sampling Zoo dependency is unavailable.')
        print('Install with: pip install "fedot[sampling_zoo]"')
        raise ex
