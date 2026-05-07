__all__ = [
    'DataReader',
    'JSONBatchLoader',
    'TSLoader',
    'TextBatchLoader',
    'get_df_from_csv',
    'read_arff_file',
]


def __getattr__(name):
    if name == 'DataReader':
        from fedot.core.data.reader.data_reader import DataReader

        return DataReader
    if name in {'JSONBatchLoader', 'TextBatchLoader'}:
        from fedot.core.data.reader.load_data import JSONBatchLoader, TextBatchLoader

        return {'JSONBatchLoader': JSONBatchLoader, 'TextBatchLoader': TextBatchLoader}[name]
    if name in {'get_df_from_csv', 'read_arff_file'}:
        from fedot.core.data.reader.tools import get_df_from_csv, read_arff_file

        return {'get_df_from_csv': get_df_from_csv, 'read_arff_file': read_arff_file}[name]
    if name == 'TSLoader':
        from fedot.core.data.reader.ucr_loader import TSLoader

        return TSLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
