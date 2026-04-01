from __future__ import annotations

from typing import Any

import pandas as pd


def dataframe_to_markdown(frame: pd.DataFrame, *, index: bool = False) -> str:
    try:
        return frame.to_markdown(index=index)
    except Exception:
        return _render_markdown_fallback(frame, index=index)


def _render_markdown_fallback(frame: pd.DataFrame, *, index: bool = False) -> str:
    normalized = frame.copy()
    if index:
        normalized = normalized.reset_index()

    columns = [str(column) for column in normalized.columns]
    rows = [columns]
    for record in normalized.itertuples(index=False, name=None):
        rows.append([_format_cell(value) for value in record])

    if not rows:
        return ""

    widths = [max(len(str(row[column_index])) for row in rows) for column_index in range(len(columns))]

    def _render_row(values: list[Any]) -> str:
        cells = [str(value).ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = _render_row(rows[0])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [_render_row(row) for row in rows[1:]]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")
