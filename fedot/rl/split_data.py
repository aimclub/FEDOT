from os import walk
from os.path import join

import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.core.utils import fedot_project_root

if __name__ == '__main__':
    path_to_datasets = join(fedot_project_root(), 'fedot/rl/data/')
    path_to_save_train = join(fedot_project_root(), 'fedot/rl/data/train/')
    path_to_save_valid = join(fedot_project_root(), 'fedot/rl/data/valid/')

    files = [filename for (_, _, filename) in walk(path_to_datasets)][0]

    for file in files:
        path_to_dataset = join(path_to_datasets, file)

        data = pd.read_csv(path_to_dataset)

        train_data, valid_data = train_test_split(data, test_size=0.3)

        train_data.to_csv(path_or_buf=join(path_to_save_train, file))
        valid_data.to_csv(path_or_buf=join(path_to_save_valid, file))
