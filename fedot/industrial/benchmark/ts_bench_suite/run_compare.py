from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ArtifactRecord, ensure_directory
from .markdown import dataframe_to_markdown


@dataclass(frozen=True)
class RegisteredRunComparison:
    registry_frame: pd.DataFrame
    best_models_frame: pd.DataFrame
    model_metric_frame: pd.DataFrame
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


class RegisteredRunComparisonError(ValueError):
    pass


def load_registry_entries(base_dir: str | Path) -> pd.DataFrame:
    registry_dir = _resolve_registry_dir(base_dir)
    rows = []
    for entry_path in sorted(registry_dir.glob('*.json')):
        rows.append(json.loads(entry_path.read_text(encoding='utf-8')))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if 'run_id' in frame.columns:
        frame = frame.sort_values('run_id').reset_index(drop=True)
    return frame


def compare_registered_runs(
        base_dir: str | Path,
        *,
        run_ids: tuple[str, ...] = (),
        output_dir: str | Path | None = None,
) -> RegisteredRunComparison:
    registry_frame = load_registry_entries(base_dir)
    if registry_frame.empty:
        raise RegisteredRunComparisonError('Registry does not contain any run entries.')

    if run_ids:
        registry_frame = registry_frame[registry_frame['run_id'].isin(run_ids)].copy()
    if registry_frame.empty:
        raise RegisteredRunComparisonError('No registry entries matched the requested run_ids.')

    if registry_frame['task_type'].nunique() > 1:
        raise RegisteredRunComparisonError('Run comparison currently expects entries from a single task type.')

    best_models_frame, model_metric_frame = _collect_run_level_tables(registry_frame)
    artifact_manifest: list[ArtifactRecord] = []

    if output_dir is not None:
        artifact_manifest.extend(
            render_registered_run_comparison_pack(
                registry_frame=registry_frame,
                best_models_frame=best_models_frame,
                model_metric_frame=model_metric_frame,
                output_dir=output_dir,
            )
        )

    return RegisteredRunComparison(
        registry_frame=registry_frame.reset_index(drop=True),
        best_models_frame=best_models_frame.reset_index(drop=True),
        model_metric_frame=model_metric_frame.reset_index(drop=True),
        artifact_manifest=tuple(artifact_manifest),
    )


def render_registered_run_comparison_pack(
        *,
        registry_frame: pd.DataFrame,
        best_models_frame: pd.DataFrame,
        model_metric_frame: pd.DataFrame,
        output_dir: str | Path,
) -> tuple[ArtifactRecord, ...]:
    target_dir = ensure_directory(output_dir)
    manifest: list[ArtifactRecord] = []

    for name, frame in (
            ('registry_runs', registry_frame),
            ('best_models', best_models_frame),
            ('model_metric_matrix', model_metric_frame),
    ):
        manifest.extend(_write_frame_bundle(frame, target_dir / name))

    summary_path = target_dir / 'comparison_summary.md'
    summary_lines = [
        '# Registered Run Comparison',
        '',
        f'- Number of runs: `{len(registry_frame)}`',
        f'- Task type: `{registry_frame["task_type"].iloc[0]}`',
        '',
        '## Run Overview',
        '',
        dataframe_to_markdown(registry_frame, index=False),
    ]
    if not best_models_frame.empty:
        summary_lines.extend(['', '## Best Models Per Run', '', dataframe_to_markdown(best_models_frame, index=False)])
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    if not best_models_frame.empty:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metric_name = str(best_models_frame['primary_metric'].iloc[0])
        chart_frame = best_models_frame.sort_values('run_id')
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(chart_frame['run_id'], chart_frame['best_metric_value'])
        axis.set_title(f'Best {metric_name.upper()} by Run')
        axis.set_xlabel('Run ID')
        axis.set_ylabel(metric_name.upper())
        axis.tick_params(axis='x', rotation=25)
        axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'best_{metric_name}_by_run.{extension}'
            figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(figure)

        if not model_metric_frame.empty and len(model_metric_frame.columns) > 1:
            heatmap_source = model_metric_frame.set_index('model_name')
            if not heatmap_source.empty:
                heatmap_figure, heatmap_axis = plt.subplots(figsize=(10, max(4, 0.5 * len(heatmap_source))))
                image = heatmap_axis.imshow(heatmap_source.to_numpy(dtype=float), aspect='auto', cmap='viridis')
                heatmap_axis.set_xticks(range(len(heatmap_source.columns)))
                heatmap_axis.set_xticklabels(heatmap_source.columns, rotation=25, ha='right')
                heatmap_axis.set_yticks(range(len(heatmap_source.index)))
                heatmap_axis.set_yticklabels(heatmap_source.index)
                heatmap_axis.set_title(f'Model Metric Matrix ({metric_name.upper()})')
                heatmap_figure.colorbar(image, ax=heatmap_axis)
                for extension in ('png', 'svg'):
                    path = target_dir / f'model_metric_matrix.{extension}'
                    heatmap_figure.savefig(path, dpi=200, bbox_inches='tight')
                    manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(heatmap_figure)

    return tuple(manifest)


def _resolve_registry_dir(base_dir: str | Path) -> Path:
    candidate = Path(base_dir)
    if candidate.name == '_registry':
        return candidate
    return candidate / '_registry'


def _collect_run_level_tables(registry_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for entry in registry_frame.to_dict(orient='records'):
        run_dir = Path(entry['run_dir'])
        primary_metric = str(entry['primary_metric'])
        leaderboard_path = run_dir / 'aggregate' / 'leaderboard.csv'
        if not leaderboard_path.exists():
            continue
        leaderboard = pd.read_csv(leaderboard_path)
        if leaderboard.empty or primary_metric not in leaderboard.columns:
            continue
        ascending = primary_metric != 'accuracy' and primary_metric != 'balanced_accuracy' and primary_metric != 'f1_macro' and primary_metric != 'r2'
        leaderboard = leaderboard.sort_values(primary_metric, ascending=ascending).reset_index(drop=True)
        best_row = leaderboard.iloc[0]
        best_rows.append(
            {
                'run_id': entry['run_id'],
                'task_type': entry['task_type'],
                'primary_metric': primary_metric,
                'best_model_name': best_row['model_name'],
                'best_metric_value': float(best_row[primary_metric]),
                'n_models': len(leaderboard),
            }
        )
        for _, row in leaderboard.iterrows():
            model_rows.append(
                {
                    'run_id': entry['run_id'],
                    'model_name': row['model_name'],
                    'primary_metric': primary_metric,
                    'metric_value': float(row[primary_metric]),
                }
            )

    best_models_frame = pd.DataFrame(best_rows)
    if not model_rows:
        return best_models_frame, pd.DataFrame(columns=['model_name'])

    model_frame = pd.DataFrame(model_rows)
    model_metric_frame = (
        model_frame.pivot_table(index='model_name', columns='run_id', values='metric_value', aggfunc='mean')
        .reset_index()
        .sort_values('model_name')
        .reset_index(drop=True)
    )
    return best_models_frame, model_metric_frame


def _write_frame_bundle(frame: pd.DataFrame, path_without_suffix: Path) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    csv_path = path_without_suffix.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    artifacts.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))

    tex_path = path_without_suffix.with_suffix('.tex')
    tex_path.write_text(frame.to_latex(index=False), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='table', path=str(tex_path), format='tex'))

    json_path = path_without_suffix.with_suffix('.json')
    json_path.write_text(frame.to_json(orient='records', indent=2), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='structured', path=str(json_path), format='json'))
    return artifacts
