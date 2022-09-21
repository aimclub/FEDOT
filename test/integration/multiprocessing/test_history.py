from test.unit.composer.test_history import test_newly_generated_history


def test_newly_generated_history_with_multiprocessing():
    test_newly_generated_history(n_jobs=2)
