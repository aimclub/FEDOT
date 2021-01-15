import os
import pathlib
import shutil
from os.path import join, abspath, basename

import pytest

from fedot.core.data.load_data import TextBatchLoader


@pytest.fixture(scope="session", autouse=True)
def create_test_data(request):
    test_data_dir = 'loader_test_data_dir'

    # create files
    created_files = []
    for dir_index in range(1, 3):
        # create subdir
        dir_name = abspath(join(test_data_dir, f'subdir{dir_index}'))
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        for file_index in range(1, 3):
            # create file in subdir
            file_name = join(dir_name,
                             f'{dir_index}_subdir_{file_index}_file.txt')

            # fill in file
            with open(file_name, 'w') as file:
                file.write(f'{basename(file_name)} content')
            created_files.append(file_name)

    request.addfinalizer(remove_test_data)


def remove_test_data():
    test_data_path = 'loader_test_data_dir'
    shutil.rmtree(test_data_path)
    os.remove('meta_loader_test_data_dir.csv')


def test_text_batch_loader():
    path = 'loader_test_data_dir'
    test_loader = TextBatchLoader(path)
    df = test_loader.extract()
    contents = sorted(df['text'].tolist())

    assert df.size == 8
    assert contents[0] == '1_subdir_1_file.txt content'
