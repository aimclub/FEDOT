import os
from abc import ABC, abstractmethod

import pandas as pd


class BatchLoader(ABC):
    """ Class for loading data with batches """
    def __init__(self, path: str):
        self.path = path
        self.meta_df = None
        self.target_name = 'label'

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        pass

    def _extract_files_paths(self):
        all_files = []
        for root, dirs, files in os.walk(self.path):
            files_paths = []
            for name in files:
                if not name.startswith('.'):
                    whole_file_path = os.path.join(root, name)
                    files_paths.append(whole_file_path)

            if files:
                all_files.extend(files_paths)
        self._load_to_meta_df(all_files)

    def _load_to_meta_df(self, files):
        data_rows = []
        for file in files:
            dir_name = os.path.basename(os.path.dirname(file))
            row = [file, dir_name]
            data_rows.append(row)

        self.meta_df = pd.DataFrame(data=data_rows,
                                    columns=['file_path', f'{self.target_name}'])
        # shuffle samples
        self.meta_df = self.meta_df.sample(frac=1).reset_index(drop=True)

    def export_to_csv(self, path: str = None):
        if not path:
            path = self.path
            export_filename = f'meta_{os.path.basename(path)}.csv'
            export_dirname = os.path.dirname(path)
            self.meta_df.to_csv(os.path.join(export_dirname, export_filename))
        else:
            self.meta_df.to_csv(os.path.abspath(path))


class TextBatchLoader(BatchLoader):
    """ Class for loading text data with batches """

    def __init__(self, path: str):
        if os.path.isfile(path):
            raise ValueError('Expected directory path but got file')
        super().__init__(path)

    def extract(self, export: bool = True):
        self._extract_files_paths()

        content_list = []
        for file_path in self.meta_df['file_path'].tolist():
            with open(file_path, 'r') as text_file:
                content = text_file.read()
                content_list.append(content)

        new_column = pd.Series(data=content_list, index=self.meta_df.index)

        self.meta_df.insert(loc=self.meta_df.columns.get_loc('file_path'),
                            column='text',
                            value=new_column)
        self.meta_df = self.meta_df.drop(['file_path'], axis=1)

        if export:
            self.export_to_csv()

        return self.meta_df
