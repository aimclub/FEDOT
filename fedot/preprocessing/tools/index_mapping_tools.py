from typing import Optional, Dict, List, Tuple

from fedot.core.data.complex_types import ArrayType, IndexType


def create_index_mapping(features: ArrayType, init_shape: Optional[Tuple[int]] = None) -> Dict[int, int]:
    """Create initial feature index mapping for preprocessing pipeline.

    Mapping format is `{current_idx: original_idx}` and is used to translate
    user-declared feature indices between preprocessing steps.

    Args:
        features: Current feature matrix/tensor.
        init_shape: Optional initial shape for time-series data. For 3D time
            series, mapping is created on logical feature channels.

    Returns:
        Initial index mapping where each current column points to its source
        column index.
    """
    if init_shape is None or len(init_shape) == 2:  # for 2-dimensional
        return {idx: idx for idx in range(features.shape[1])}
    else:  # for 3-dimensional, only for ts type
        return {idx: idx for idx in range(features.shape[1] // init_shape[2])}


def update_index_mapping(
    index_mapping: Dict[int, int],
    changed_idx: IndexType,
    features: ArrayType,
    new_cols_dict: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Update `{current_idx: original_idx}` mapping after a preprocessing step.

    Main rule for index consistency:
    if a step expands selected columns into multiple columns, original selected
    columns are considered removed from current feature space, and produced
    columns are appended to the end of feature matrix. New appended columns keep
    link to the same original source indices.

    Args:
        index_mapping: Previous mapping in `{current_idx: original_idx}` format.
        changed_idx: Current indices of columns that were transformed by step.
        features: Feature matrix after step execution.
        new_cols_dict: Optional explicit mapping `{changed_current_idx: n_new_cols}`
            for steps that expand columns unevenly.

    Returns:
        Updated index mapping aligned with post-step feature layout.
    """
    if index_mapping is None:
        return None

    if changed_idx is None:
        return index_mapping

    changed_idx = sorted(set(changed_idx))
    old_n_cols = max(index_mapping.keys()) + 1
    new_n_cols = features.shape[1]

    if old_n_cols == new_n_cols:
        return dict(index_mapping)

    if old_n_cols > new_n_cols:
        remaining_original_idx = [
            original_idx
            for current_idx, original_idx in sorted(index_mapping.items())
            if current_idx not in changed_idx
        ]

        if len(remaining_original_idx) < new_n_cols:
            raise ValueError(
                f"Too many columns were removed: got {len(remaining_original_idx)} "
                f"remaining columns for features with {new_n_cols} columns."
            )

        remaining_original_idx = remaining_original_idx[:new_n_cols]
        return {
            new_idx: original_idx
            for new_idx, original_idx in enumerate(remaining_original_idx)
        }

    # old_n_cols < new_n_cols
    remaining_items = {
        current_idx: original_idx
        for current_idx, original_idx in index_mapping.items()
        if current_idx not in changed_idx
    }

    updated_mapping: Dict[int, int] = {}
    for current_idx in sorted(remaining_items.keys()):
        shift = sum(1 for removed_idx in changed_idx if removed_idx < current_idx)
        updated_mapping[current_idx - shift] = remaining_items[current_idx]

    appended_total = new_n_cols - len(updated_mapping)
    if appended_total < 0:
        raise ValueError(
            f"Invalid shapes: resulting number of columns ({new_n_cols}) is smaller "
            f"than remaining mapping size ({len(updated_mapping)})."
        )

    if new_cols_dict is None:
        if len(changed_idx) == 0:
            if appended_total != 0:
                raise ValueError("No changed_idx provided, but number of columns changed.")
            per_col_counts = {}
        else:
            if appended_total % len(changed_idx) != 0:
                raise ValueError(
                    "Cannot distribute new columns equally across changed_idx. "
                    "Pass new_cols_dict explicitly."
                )
            per_col = appended_total // len(changed_idx)
            per_col_counts = {idx: per_col for idx in changed_idx}
    else:
        per_col_counts = dict(new_cols_dict)
        if set(per_col_counts.keys()) != set(changed_idx):
            raise ValueError("new_cols_dict keys must exactly match changed_idx.")
        if sum(per_col_counts.values()) != appended_total:
            raise ValueError(
                f"Sum of new_cols_dict values ({sum(per_col_counts.values())}) "
                f"must equal number of appended columns ({appended_total})."
            )

    next_idx = len(updated_mapping)
    for changed_current_idx in changed_idx:
        original_idx = index_mapping[changed_current_idx]
        n_new_cols = per_col_counts[changed_current_idx]

        for _ in range(n_new_cols):
            updated_mapping[next_idx] = original_idx
            next_idx += 1

    return updated_mapping


def update_indices(index_mapping: Dict[int, int], indices: IndexType) -> List[int]:
    """
    Convert original feature indices to current indices for next step execution.

    Args:
        index_mapping (Dict[int, int]): Mapping in format `{current_idx: original_idx}`.
        indices (IndexType): Original/source feature indices to update.

    Returns:
        List[int]: Current feature indices.
    """
    if indices is None or index_mapping is None:
        return indices

    updated_indices = []

    for old_idx in indices:
        matched_new_indices = [
            new_idx for new_idx, mapped_old_idx in index_mapping.items()
            if mapped_old_idx == old_idx
        ]

        if not matched_new_indices:
            raise ValueError(f"Old index {old_idx} is not present in index_mapping.")

        updated_indices.append(max(matched_new_indices))

    return updated_indices


def agregate_idx_from_step(steps: List):
    """Collect feature indices from all preprocessing steps.

    Args:
        steps: Sequence of preprocessing steps with `features_idx`.

    Returns:
        Flat list of feature indices used by the provided steps.
    """
    features_idx = []
    for step in steps:
        features_idx.extend(step.features_idx)

    return features_idx
