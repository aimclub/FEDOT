"""Resolve optional selectors without confusing source and prepared columns."""
from fedot.core.data.tensor_data.contracts import as_list, column_indices
from fedot.core.data.tensor_data.tools import get_idx_from_features_names


def resolve_optional_columns(data, selectors) -> list[int]:
    """Return columns in the optional service's input coordinate system.

    Source names select all their expanded columns. A generated prepared name
    selects exactly one column. Numeric selectors retain legacy source-position
    semantics. Containers without fitted schema keep legacy name resolution,
    while expanded sources consistently select all their prepared columns.
    """
    selected = as_list(selectors)
    state = getattr(data, 'preparation_state', None)
    if state is None or state.schema.names is None or not selected or not all(
            isinstance(name, str) for name in selected):
        positions = get_idx_from_features_names(selectors, data.features_names)
        if positions is None or data.idx_mapping is None:
            return positions
        columns = []
        for source in positions:
            matches = [pos for pos, original in sorted(data.idx_mapping.items())
                       if original == source]
            if not matches:
                raise ValueError(
                    f'Old index {source} is not present in index_mapping.')
            columns.extend(matches)
        return list(dict.fromkeys(columns))

    sources = {name: source for (_, source), name in zip(
        state.schema.mapping, state.schema.names)}
    columns = []
    for name in selected:
        if name in sources:
            columns.extend(pos for pos, source in sorted(data.idx_mapping.items())
                           if source == sources[name])
        else:
            columns.extend(column_indices(
                [name], data.features.shape[1], data.features_names, 'features_idx'))
    return list(dict.fromkeys(columns))


def compose_source_mapping(parent: dict[int, int],
                           transformed: dict[int, int]) -> dict[int, int]:
    """Compose optional-input ownership with original source ownership."""
    return {pos: parent.get(source, source) for pos, source in transformed.items()}
