import tracemalloc


class MemoryAnalytics:
    is_active = False

    @classmethod
    def start(cls):
        tracemalloc.start()
        cls.is_active = True

    @classmethod
    def finish(cls):
        tracemalloc.stop()
        cls.is_active = False
        cls.log(cls, additional_info='finish')

    @classmethod
    def log(cls, logger=None, additional_info: str = 'location'):
        if cls.is_active:
            memory_consumption = tracemalloc.get_traced_memory()
            message = f'Memory consumption for {additional_info}: ' \
                      f'current {round(memory_consumption[0] / 1024 / 1024, 1)} Mb, ' \
                      f'max: {round(memory_consumption[1] / 1024 / 1024, 1)} Mb'
            if logger is not None:
                logger.message(message)
            else:
                print(message)
