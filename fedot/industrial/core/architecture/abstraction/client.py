from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels


def use_default_fedot_client(func):
    def decorated_func(self, *args, **kwargs):
        repo = IndustrialModels()
        repo.setup_default_repository()
        result = func(self, *args, **kwargs)
        repo.setup_repository()
        return result

    return decorated_func


def use_industrial_fedot_client(func):
    def decorated_func(self, *args):
        repo = IndustrialModels()
        result = func(self, *args)
        repo.setup_repository()
        return result

    return decorated_func
