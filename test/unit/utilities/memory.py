from fedot.utilities.memory import MemoryAnalytics


def test_memory_logging_in_active_mode():
    MemoryAnalytics.start()
    message = MemoryAnalytics.log(additional_info='test')
    current, maximal = MemoryAnalytics.get_measures()
    MemoryAnalytics.finish()

    assert 'test' in message
    assert current > 0 and maximal > 0


def test_memory_logging_in_non_active_mode():
    message = MemoryAnalytics.log(additional_info='test')
    assert len(message) == 0
