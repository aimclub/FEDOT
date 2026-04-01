from __future__ import annotations

from pathlib import Path

import pandas as pd

from .core import LoadedTabularDataset, TabularDatasetSpec, resolve_size_bucket


ROOT_DIR = Path(__file__).resolve().parents[4]


DATASET_REGISTRY: dict[str, TabularDatasetSpec] = {
    'kc2': TabularDatasetSpec(
        name='kc2',
        problem='classification',
        loader_kind='local_csv',
        path='examples/real_cases/data/kc2/kc2.csv',
        target_column='problems',
    ),
    'amlb_spambase': TabularDatasetSpec(
        name='amlb_spambase',
        problem='classification',
        loader_kind='openml',
        loader_params={'openml_name': 'spambase'},
    ),
    'scoring_train': TabularDatasetSpec(
        name='scoring_train',
        problem='classification',
        loader_kind='local_csv',
        path='examples/real_cases/data/scoring/scoring_train.csv',
        target_column='target',
        loader_params={'drop_columns': ('ID',)},
    ),
    'cholesterol': TabularDatasetSpec(
        name='cholesterol',
        problem='regression',
        loader_kind='local_csv',
        path='examples/real_cases/data/cholesterol/cholesterol.csv',
        target_column='target',
    ),
    'diabetes': TabularDatasetSpec(
        name='diabetes',
        problem='regression',
        loader_kind='sklearn',
        loader_params={'dataset_name': 'diabetes'},
    ),
    'california_housing': TabularDatasetSpec(
        name='california_housing',
        problem='regression',
        loader_kind='sklearn',
        loader_params={'dataset_name': 'california_housing'},
    ),
    'make_classification_small': TabularDatasetSpec(
        name='make_classification_small',
        problem='classification',
        loader_kind='synthetic_classification',
        loader_params={
            'n_samples': 600,
            'n_features': 20,
            'n_informative': 10,
            'n_redundant': 4,
            'n_classes': 2,
        },
    ),
    'make_regression_medium': TabularDatasetSpec(
        name='make_regression_medium',
        problem='regression',
        loader_kind='synthetic_regression',
        loader_params={
            'n_samples': 5000,
            'n_features': 30,
            'n_informative': 12,
            'noise': 0.2,
        },
    ),
    'make_regression_large': TabularDatasetSpec(
        name='make_regression_large',
        problem='regression',
        loader_kind='synthetic_regression',
        loader_params={
            'n_samples': 15000,
            'n_features': 40,
            'n_informative': 16,
            'noise': 0.4,
        },
    ),
}


def resolve_dataset_specs(dataset_names: tuple[str, ...]) -> tuple[TabularDatasetSpec, ...]:
    resolved_specs = []
    for dataset_name in dataset_names:
        normalized_name = dataset_name.strip().lower()
        if normalized_name not in DATASET_REGISTRY:
            raise ValueError(f'Unsupported benchmark dataset: {dataset_name}')
        resolved_specs.append(DATASET_REGISTRY[normalized_name])
    return tuple(resolved_specs)


def load_dataset(spec: TabularDatasetSpec, seed: int = 42) -> LoadedTabularDataset:
    if spec.loader_kind == 'local_csv':
        return _load_local_csv_dataset(spec)
    if spec.loader_kind == 'openml':
        return _load_openml_dataset(spec)
    if spec.loader_kind == 'sklearn':
        return _load_sklearn_dataset(spec)
    if spec.loader_kind == 'synthetic_classification':
        return _load_synthetic_classification_dataset(spec, seed=seed)
    if spec.loader_kind == 'synthetic_regression':
        return _load_synthetic_regression_dataset(spec, seed=seed)
    raise ValueError(f'Unsupported dataset loader kind: {spec.loader_kind}')


def _load_local_csv_dataset(spec: TabularDatasetSpec) -> LoadedTabularDataset:
    dataset_path = ROOT_DIR / str(spec.path)
    frame = pd.read_csv(dataset_path)
    target_column = spec.target_column
    if target_column is None:
        raise ValueError(f'Dataset {spec.name} requires target_column for local_csv loader.')

    features = frame.drop(columns=[target_column, *spec.loader_params.get('drop_columns', ())], errors='ignore')
    target = frame[target_column]
    return _build_loaded_dataset(spec, features, target, metadata={'path': str(dataset_path)})


def _load_openml_dataset(spec: TabularDatasetSpec) -> LoadedTabularDataset:
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(
        name=spec.loader_params['openml_name'],
        as_frame=True,
        parser='auto',
    )
    features = dataset.data
    target = dataset.target
    metadata = {
        'openml_name': spec.loader_params['openml_name'],
        'frame_shape': tuple(features.shape),
    }
    return _build_loaded_dataset(spec, features, target, metadata=metadata)


def _load_sklearn_dataset(spec: TabularDatasetSpec) -> LoadedTabularDataset:
    from sklearn.datasets import fetch_california_housing, load_diabetes

    dataset_name = spec.loader_params['dataset_name']
    if dataset_name == 'diabetes':
        bundle = load_diabetes(as_frame=True)
    elif dataset_name == 'california_housing':
        bundle = fetch_california_housing(as_frame=True)
    else:
        raise ValueError(f'Unsupported sklearn dataset: {dataset_name}')

    features = bundle.data
    target = bundle.target
    return _build_loaded_dataset(
        spec,
        features,
        target,
        metadata={'dataset_name': dataset_name},
    )


def _load_synthetic_classification_dataset(spec: TabularDatasetSpec, seed: int) -> LoadedTabularDataset:
    from sklearn.datasets import make_classification

    generator_params = dict(spec.loader_params)
    generator_params['random_state'] = seed
    features, target = make_classification(**generator_params)
    features = pd.DataFrame(features)
    target = pd.Series(target)
    return _build_loaded_dataset(spec, features, target, metadata={'random_state': seed})


def _load_synthetic_regression_dataset(spec: TabularDatasetSpec, seed: int) -> LoadedTabularDataset:
    from sklearn.datasets import make_regression

    generator_params = dict(spec.loader_params)
    generator_params['random_state'] = seed
    features, target = make_regression(**generator_params)
    features = pd.DataFrame(features)
    target = pd.Series(target)
    return _build_loaded_dataset(spec, features, target, metadata={'random_state': seed})


def _build_loaded_dataset(spec: TabularDatasetSpec, features, target, metadata: dict) -> LoadedTabularDataset:
    sample_count = int(len(target))
    feature_count = int(features.shape[1])
    return LoadedTabularDataset(
        spec=spec,
        features=features,
        target=target,
        sample_count=sample_count,
        feature_count=feature_count,
        size_bucket=resolve_size_bucket(sample_count),
        metadata=metadata,
    )

