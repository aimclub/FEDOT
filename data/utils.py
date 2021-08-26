import os
import shutil


def get_simple_composer_params() -> dict:
    params = {'max_depth': 2,
              'max_arity': 3,
              'pop_size': 2,
              'num_of_generations': 2,
              'timeout': 1,
              'preset': 'ultra_light'}
    return params


def create_func_delete_files(paths):
    """
    Create function to delete pipelines.
    """

    def wrapper():
        for path in paths:
            path = create_correct_path(path, True)
            if path is not None and os.path.isdir(path):
                shutil.rmtree(path)

    return wrapper


def create_correct_path(path: str, dirname_flag: bool = False):
    """
    Create path with time which was created during the testing process.
    """

    for dirname in next(os.walk(os.path.curdir))[1]:
        if dirname.endswith(path):
            if dirname_flag:
                return dirname
            else:
                file = os.path.join(dirname, path + '.json')
                return file
    return None
