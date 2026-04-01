from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / 'fedot_ind' / 'data'


class LocalDatasetParseError(ValueError):
    pass


@dataclass(frozen=True)
class LocalSplitData:
    train_features: np.ndarray
    train_target: np.ndarray
    test_features: np.ndarray
    test_target: np.ndarray
    metadata: dict[str, Any]


def resolve_local_split_paths(
        dataset_name: str,
        *,
        data_root: str | Path | None = None,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
) -> tuple[Path, Path]:
    if train_path is not None and test_path is not None:
        return Path(train_path), Path(test_path)

    dataset_dir = Path(data_root or DEFAULT_LOCAL_DATA_DIR) / dataset_name
    if not dataset_dir.exists():
        raise LocalDatasetParseError(f'Local dataset directory does not exist: {dataset_dir}')

    train_base = dataset_dir / f'{dataset_name}_TRAIN'
    test_base = dataset_dir / f'{dataset_name}_TEST'
    for extension in ('.tsv', '.csv', '.ts'):
        train_candidate = train_base.with_suffix(extension)
        test_candidate = test_base.with_suffix(extension)
        if train_candidate.exists() and test_candidate.exists():
            return train_candidate, test_candidate

    raise LocalDatasetParseError(f'Could not resolve local TRAIN/TEST files for {dataset_name} in {dataset_dir}')


def load_local_supervised_split(
        dataset_name: str,
        *,
        data_root: str | Path | None = None,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
) -> LocalSplitData:
    train_file, test_file = resolve_local_split_paths(
        dataset_name,
        data_root=data_root,
        train_path=train_path,
        test_path=test_path,
    )

    if train_file.suffix != test_file.suffix:
        raise LocalDatasetParseError('TRAIN/TEST files use different suffixes.')

    suffix = train_file.suffix.lower()
    if suffix == '.tsv':
        train_features, train_target = _load_tabular_split(train_file, separator='\t')
        test_features, test_target = _load_tabular_split(test_file, separator='\t')
        split_provenance = 'local_tsv'
    elif suffix == '.csv':
        train_features, train_target = _load_tabular_split(train_file, separator=',')
        test_features, test_target = _load_tabular_split(test_file, separator=',')
        split_provenance = 'local_csv'
    elif suffix == '.ts':
        train_features, train_target, parser_metadata = _load_ts_split(train_file)
        test_features, test_target, test_metadata = _load_ts_split(test_file)
        parser_metadata.update(
            {
                key: value
                for key, value in test_metadata.items()
                if key not in parser_metadata
            }
        )
        split_provenance = 'local_ts'
    else:  # pragma: no cover
        raise LocalDatasetParseError(f'Unsupported local split suffix: {suffix}')

    metadata = {
        'source_train_file': train_file.name,
        'source_test_file': test_file.name,
        'source_format': suffix.lstrip('.'),
        'split_provenance': split_provenance,
    }
    if suffix == '.ts':
        metadata.update(parser_metadata)

    return LocalSplitData(
        train_features=np.asarray(train_features, dtype=float),
        train_target=np.asarray(train_target),
        test_features=np.asarray(test_features, dtype=float),
        test_target=np.asarray(test_target),
        metadata=metadata,
    )


def _load_tabular_split(path: Path, *, separator: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=separator, header=None)
    if frame.shape[1] < 2:
        raise LocalDatasetParseError(f'Tabular split file must contain target and at least one feature: {path}')
    target = frame.iloc[:, 0].to_numpy()
    features = frame.iloc[:, 1:].to_numpy(dtype=float)
    return features, target


def _load_ts_split(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    timestamps = False
    target_label = False
    class_label = False
    dimensions = None
    metadata: dict[str, Any] = {}
    data_started = False
    case_vectors: list[np.ndarray] = []
    targets: list[Any] = []

    with path.open('r', encoding='utf-8') as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            lower = line.lower()
            if not data_started:
                if lower.startswith('@timestamps'):
                    timestamps = _parse_boolean_tag(line)
                elif lower.startswith('@targetlabel'):
                    target_label = _parse_boolean_tag(line)
                elif lower.startswith('@classlabel'):
                    class_label = _parse_boolean_tag(line)
                elif lower.startswith('@dimensions'):
                    parts = line.split()
                    if len(parts) >= 2:
                        dimensions = int(parts[1])
                elif lower.startswith('@problemname'):
                    parts = line.split(maxsplit=1)
                    metadata['problem_name'] = parts[1] if len(parts) > 1 else path.stem
                elif lower.startswith('@univariate'):
                    metadata['univariate'] = _parse_boolean_tag(line)
                elif lower.startswith('@equallength') or lower.startswith('@equallength'):
                    metadata['equal_length'] = _parse_boolean_tag(line)
                elif lower.startswith('@data'):
                    data_started = True
                continue

            if timestamps:
                raise LocalDatasetParseError(f'Timestamped .ts files are not supported yet: {path}')
            if not target_label and not class_label:
                raise LocalDatasetParseError(f'.ts file must define @targetlabel or @classlabel: {path}')

            split_index = line.rfind(':')
            if split_index < 0:
                raise LocalDatasetParseError(f'Could not split series and target in line from {path}')

            series_part = line[:split_index]
            target_part = line[split_index + 1:].strip()
            dimension_parts = series_part.split(':')
            if dimensions is not None and len(dimension_parts) != dimensions:
                raise LocalDatasetParseError(
                    f'Unexpected number of dimensions in {path}: expected {dimensions}, got {len(dimension_parts)}'
                )

            flattened: list[float] = []
            for dimension_part in dimension_parts:
                cleaned = dimension_part.strip()
                if not cleaned:
                    continue
                flattened.extend(float(token) for token in cleaned.split(',') if token)

            case_vectors.append(np.asarray(flattened, dtype=float))
            if target_label:
                targets.append(float(target_part))
            else:
                targets.append(target_part)

    if not case_vectors:
        raise LocalDatasetParseError(f'No cases were parsed from .ts file: {path}')

    lengths = {vector.size for vector in case_vectors}
    if len(lengths) != 1:
        raise LocalDatasetParseError(
            f'Variable-length .ts records are not supported by the flat benchmark-v2 adapters yet: {path}'
        )

    metadata['dimensions'] = dimensions or 1
    metadata['target_type'] = 'regression' if target_label else 'classification'
    features = np.vstack(case_vectors)
    target_array = np.asarray(targets, dtype=float if target_label else object)
    return features, target_array, metadata


def _parse_boolean_tag(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        raise LocalDatasetParseError(f'Malformed boolean metadata tag: {line}')
    return parts[1].strip().lower() == 'true'
