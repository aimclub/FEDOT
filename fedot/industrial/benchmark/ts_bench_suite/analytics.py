from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    ArtifactRecord,
    ForecastingBenchmarkResult,
    MetricRecord,
    PredictionRecord,
    RunStatus,
    ensure_directory,
    to_plain_data,
    write_json,
)
from .markdown import dataframe_to_markdown


@dataclass(frozen=True)
class SeriesComparisonResult:
    series_id: str
    dataset_name: str
    model_names: tuple[str, ...]
    metrics_table: pd.DataFrame
    prediction_table: pd.DataFrame
    artifact_manifest: tuple[ArtifactRecord, ...] = ()


def _series_record_lookup(result: ForecastingBenchmarkResult) -> dict[str, Any]:
    return {record.series_id: record for record in result.series_records}


def _slugify_model_name(name: str) -> str:
    return ''.join(character.lower() if character.isalnum() else '_' for character in name).strip('_')


def predictions_to_frame(records: tuple[PredictionRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


def metrics_to_frame(records: tuple[MetricRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame([to_plain_data(record) for record in records])


def runs_to_frame(result: ForecastingBenchmarkResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in result.run_records:
        row = {
            'run_id': record.run_id,
            'benchmark': record.benchmark,
            'dataset_name': record.dataset_name,
            'subset': record.subset,
            'series_id': record.series_id,
            'model_name': record.model_name,
            'status': record.status.value,
            'message': record.message,
        }
        row.update(record.metrics_summary)
        rows.append(row)
    return pd.DataFrame(rows)


def build_benchmark_leaderboard(
        result: ForecastingBenchmarkResult,
        primary_metric: str | None = None,
) -> pd.DataFrame:
    metric_name = primary_metric or result.aggregate_report.primary_metric
    run_frame = runs_to_frame(result)
    if run_frame.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    successful = run_frame[run_frame['status'] == RunStatus.SUCCESS.value]
    if successful.empty:
        return pd.DataFrame(columns=['benchmark', 'dataset_name', 'model_name', metric_name, 'n_series', 'rank'])
    leaderboard = (
        successful.groupby(['benchmark', 'dataset_name', 'model_name'])[metric_name]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': metric_name, 'count': 'n_series'})
        .sort_values(metric_name)
        .reset_index(drop=True)
    )
    leaderboard['rank'] = leaderboard[metric_name].rank(method='dense')
    return leaderboard


def _stable_write_table(frame: pd.DataFrame, path_without_suffix: Path) -> list[ArtifactRecord]:
    artifacts: list[ArtifactRecord] = []
    csv_path = path_without_suffix.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    artifacts.append(ArtifactRecord(kind='table', path=str(csv_path), format='csv'))

    tex_path = path_without_suffix.with_suffix('.tex')
    tex_path.write_text(frame.to_latex(index=False, float_format=lambda value: f'{value:.4f}'), encoding='utf-8')
    artifacts.append(ArtifactRecord(kind='table', path=str(tex_path), format='tex'))

    parquet_path = path_without_suffix.with_suffix('.parquet')
    try:
        frame.to_parquet(parquet_path, index=False)
        artifacts.append(ArtifactRecord(kind='structured', path=str(parquet_path), format='parquet'))
    except Exception:
        pass
    return artifacts


def compare_models_on_series(
        result: ForecastingBenchmarkResult,
        series_id: str,
        output_dir: str | Path | None = None,
) -> SeriesComparisonResult:
    predictions = predictions_to_frame(result.prediction_records)
    metrics = metrics_to_frame(result.metric_records)

    series_predictions = predictions[predictions['series_id'] == series_id].copy()
    if series_predictions.empty:
        raise ValueError(f'No prediction records found for series_id={series_id}.')

    dataset_name = str(series_predictions['dataset_name'].iloc[0])
    series_lookup = _series_record_lookup(result)
    series_record = series_lookup.get(series_id)
    series_metrics = metrics[(metrics['series_id'] == series_id) & (metrics['horizon_index'].isna())].copy()
    series_metrics = series_metrics.sort_values(['metric_name', 'metric_value', 'model_name'])
    series_predictions = series_predictions.sort_values(['model_name', 'horizon_index'])

    artifact_manifest: list[ArtifactRecord] = []
    if output_dir is not None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        target_dir = ensure_directory(output_dir)
        pivot = series_predictions.pivot(index='horizon_index', columns='model_name', values='y_pred')
        truth = (
            series_predictions[['horizon_index', 'y_true']]
            .drop_duplicates()
            .sort_values('horizon_index')
            .set_index('horizon_index')
        )
        train_values = np.asarray(series_record.train_values, dtype=float) if series_record is not None else np.array(
            [])
        actual_test = np.asarray(series_record.test_values, dtype=float) if series_record is not None else truth[
            'y_true'].to_numpy()
        forecast_index = np.arange(len(train_values), len(train_values) + len(actual_test))
        history_index = np.arange(len(train_values))
        zoom_width = max(len(actual_test) * 3, min(len(train_values), 48))
        zoom_start = max(0, len(train_values) - zoom_width)

        overlay_figure, overlay_axis = plt.subplots(figsize=(12, 6))
        if len(train_values):
            overlay_axis.plot(history_index, train_values, label='train', linewidth=2.0, color='0.45')
        overlay_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
        for model_name in pivot.columns:
            overlay_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
        overlay_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        overlay_axis.set_title(f'History and Forecast Comparison for {series_id}')
        overlay_axis.set_xlabel('Time Index')
        overlay_axis.set_ylabel('Value')
        overlay_axis.legend(frameon=False)
        overlay_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_history_forecast_overlay.{extension}'
            overlay_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(overlay_figure)

        boundary_figure, boundary_axis = plt.subplots(figsize=(12, 6))
        if len(train_values):
            boundary_axis.plot(
                np.arange(zoom_start, len(train_values)),
                train_values[zoom_start:],
                label='train',
                linewidth=2.0,
                color='0.45',
            )
        boundary_axis.plot(forecast_index, actual_test, label='actual', linewidth=2.5, color='black')
        for model_name in pivot.columns:
            boundary_axis.plot(forecast_index, pivot[model_name].to_numpy(), label=model_name, linewidth=1.8)
        boundary_axis.axvline(len(train_values) - 1, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        boundary_axis.set_title(f'Boundary Zoom for {series_id}')
        boundary_axis.set_xlabel('Time Index')
        boundary_axis.set_ylabel('Value')
        boundary_axis.legend(frameon=False, ncol=2)
        boundary_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_boundary_zoom.{extension}'
            boundary_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(boundary_figure)

        delta_figure, delta_axis = plt.subplots(figsize=(10, 5))
        last_train_value = float(train_values[-1]) if len(train_values) else 0.0
        labels = ['actual'] + list(pivot.columns)
        deltas = [float(actual_test[0] - last_train_value)] if len(actual_test) else [0.0]
        deltas.extend(float(pivot[model_name].iloc[0] - last_train_value) for model_name in pivot.columns)
        delta_axis.bar(labels, deltas, color=['black'] + [f'C{index % 10}' for index in range(len(pivot.columns))])
        delta_axis.axhline(0.0, color='0.4', linestyle='--', linewidth=1.0)
        delta_axis.set_title(f'First-Step Forecast Delta for {series_id}')
        delta_axis.set_xlabel('Series / Model')
        delta_axis.set_ylabel('Delta from Last Train Value')
        delta_axis.grid(alpha=0.2, axis='y')
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_forecast_delta.{extension}'
            delta_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(delta_figure)

        residual_figure, residual_axis = plt.subplots(figsize=(9, 5))
        for model_name in pivot.columns:
            residual_axis.plot(
                forecast_index,
                actual_test - pivot[model_name].to_numpy(),
                label=model_name,
                linewidth=1.6,
            )
        residual_axis.axhline(0.0, color='black', linestyle='--', linewidth=1)
        residual_axis.set_title(f'Residuals for {series_id}')
        residual_axis.set_xlabel('Horizon')
        residual_axis.set_ylabel('Residual')
        residual_axis.legend(frameon=False)
        residual_axis.grid(alpha=0.2)
        for extension in ('png', 'svg'):
            path = target_dir / f'{series_id}_residuals.{extension}'
            residual_figure.savefig(path, dpi=200, bbox_inches='tight')
            artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(residual_figure)

        horizon_metrics = metrics[
            (metrics['series_id'] == series_id)
            & (metrics['metric_name'] == result.aggregate_report.primary_metric)
            & (metrics['horizon_index'].notna())
            ].copy()
        if not horizon_metrics.empty:
            horizon_figure, horizon_axis = plt.subplots(figsize=(9, 5))
            for model_name, group in horizon_metrics.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon Error Profile for {series_id}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(result.aggregate_report.primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = target_dir / f'{series_id}_horizon_profile.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

        okhs_records = [
            record for record in result.run_records
            if record.series_id == series_id and record.status is RunStatus.SUCCESS
               and 'okhs' in record.model_name.lower()
        ]
        if okhs_records:
            diagnostics_payload = {
                record.model_name: record.metadata for record in okhs_records
            }
            diagnostics_path = target_dir / f'{series_id}_okhs_diagnostics.json'
            write_json(diagnostics_path, diagnostics_payload)
            artifact_manifest.append(ArtifactRecord(kind='structured', path=str(diagnostics_path), format='json'))

            for record in okhs_records:
                fit_diagnostics = record.metadata.get('fdmd_fit_diagnostics', {})
                if not fit_diagnostics:
                    continue
                eigen_real = np.asarray(fit_diagnostics.get('eigenvalues_real', []), dtype=float)
                eigen_imag = np.asarray(fit_diagnostics.get('eigenvalues_imag', []), dtype=float)
                mode_norms = np.asarray(fit_diagnostics.get('mode_norms', []), dtype=float)
                prediction_diagnostics = record.metadata.get('fdmd_prediction_diagnostics', {})
                model_slug = _slugify_model_name(record.model_name)

                mode_figure, mode_axes = plt.subplots(1, 2, figsize=(12, 5))
                if len(eigen_real):
                    mode_axes[0].scatter(eigen_real, eigen_imag, alpha=0.85)
                mode_axes[0].axvline(0.0, color='0.5', linestyle='--', linewidth=1.0)
                mode_axes[0].set_title(f'Eigenvalues: {record.model_name}')
                mode_axes[0].set_xlabel('Re(lambda)')
                mode_axes[0].set_ylabel('Im(lambda)')
                mode_axes[0].grid(alpha=0.2)

                if len(mode_norms):
                    mode_axes[1].bar(np.arange(len(mode_norms)), mode_norms)
                discontinuity = prediction_diagnostics.get('boundary_discontinuity_abs_mean')
                resolved_modes = fit_diagnostics.get('resolved_n_modes')
                mode_axes[1].set_title(
                    f'Mode Norms (resolved={resolved_modes}, jump={discontinuity})'
                )
                mode_axes[1].set_xlabel('Mode Index')
                mode_axes[1].set_ylabel('Norm')
                mode_axes[1].grid(alpha=0.2, axis='y')

                for extension in ('png', 'svg'):
                    path = target_dir / f'{series_id}_{model_slug}_okhs_modes.{extension}'
                    mode_figure.savefig(path, dpi=200, bbox_inches='tight')
                    artifact_manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(mode_figure)

    return SeriesComparisonResult(
        series_id=series_id,
        dataset_name=dataset_name,
        model_names=tuple(sorted(series_predictions['model_name'].unique())),
        metrics_table=series_metrics.reset_index(drop=True),
        prediction_table=series_predictions.reset_index(drop=True),
        artifact_manifest=tuple(artifact_manifest),
    )


def render_publication_pack(
        result: ForecastingBenchmarkResult,
        output_dir: str | Path | None = None,
) -> tuple[ArtifactRecord, ...]:
    target_dir = ensure_directory(output_dir or Path(result.config.artifact_spec.output_dir) / result.run_id)
    aggregate_dir = ensure_directory(target_dir / 'aggregate')
    series_dir = ensure_directory(target_dir / 'series')

    manifest: list[ArtifactRecord] = []
    metrics_frame = metrics_to_frame(result.metric_records)
    predictions_frame = predictions_to_frame(result.prediction_records)
    runs_frame = runs_to_frame(result)
    leaderboard = build_benchmark_leaderboard(result)

    for base_name, frame in (
            ('metrics', metrics_frame),
            ('predictions', predictions_frame),
            ('runs', runs_frame),
            ('leaderboard', leaderboard),
    ):
        manifest.extend(_stable_write_table(frame, aggregate_dir / base_name))

    metadata_path = aggregate_dir / 'run_metadata.json'
    metadata_payload = {
        'run_id': result.run_id,
        'task_type': result.config.task_type.value,
        'primary_metric': result.aggregate_report.primary_metric,
        'status_counts': result.aggregate_report.status_counts,
        'dataset_specs': [to_plain_data(spec) for spec in result.config.datasets],
        'model_specs': [to_plain_data(spec) for spec in result.config.models],
    }
    write_json(metadata_path, metadata_payload)
    manifest.append(ArtifactRecord(kind='structured', path=str(metadata_path), format='json'))

    summary_path = aggregate_dir / 'summary.md'
    summary_lines = [
        f'# Forecasting Benchmark Summary: {result.run_id}',
        '',
        f'- Primary metric: `{result.aggregate_report.primary_metric}`',
        f'- Successful runs: `{result.aggregate_report.status_counts.get("success", 0)}`',
        f'- Failed runs: `{result.aggregate_report.status_counts.get("failed", 0)}`',
        f'- Skipped runs: `{result.aggregate_report.status_counts.get("skipped", 0)}`',
        f'- Not available runs: `{result.aggregate_report.status_counts.get("not_available", 0)}`',
        '',
        '## Leaderboard',
        '',
    ]
    if leaderboard.empty:
        summary_lines.append('No successful benchmark runs were recorded.')
    else:
        summary_lines.append(dataframe_to_markdown(leaderboard, index=False))
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    manifest.append(ArtifactRecord(kind='summary', path=str(summary_path), format='md'))

    successful_runs = runs_frame[runs_frame['status'] == RunStatus.SUCCESS.value].copy()
    if not successful_runs.empty:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        primary_metric = result.aggregate_report.primary_metric
        boxplot_figure, boxplot_axis = plt.subplots(figsize=(10, 5))
        successful_runs.boxplot(column=primary_metric, by='model_name', ax=boxplot_axis)
        boxplot_axis.set_title(f'{primary_metric.upper()} Distribution by Model')
        boxplot_axis.set_xlabel('Model')
        boxplot_axis.set_ylabel(primary_metric.upper())
        boxplot_axis.grid(alpha=0.2)
        boxplot_axis.figure.suptitle('')
        for extension in ('png', 'svg'):
            path = aggregate_dir / f'{primary_metric}_distribution.{extension}'
            boxplot_figure.savefig(path, dpi=200, bbox_inches='tight')
            manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
        plt.close(boxplot_figure)

        horizon_metrics = metrics_frame[
            (metrics_frame['metric_name'] == primary_metric) & (metrics_frame['horizon_index'].notna())
            ].copy()
        if not horizon_metrics.empty:
            horizon_plot = (
                horizon_metrics.groupby(['model_name', 'horizon_index'])['metric_value']
                .mean()
                .reset_index()
            )
            horizon_figure, horizon_axis = plt.subplots(figsize=(10, 5))
            for model_name, group in horizon_plot.groupby('model_name'):
                horizon_axis.plot(group['horizon_index'], group['metric_value'], label=model_name, linewidth=1.8)
            horizon_axis.set_title(f'Horizon vs {primary_metric.upper()}')
            horizon_axis.set_xlabel('Horizon')
            horizon_axis.set_ylabel(primary_metric.upper())
            horizon_axis.legend(frameon=False)
            horizon_axis.grid(alpha=0.2)
            for extension in ('png', 'svg'):
                path = aggregate_dir / f'horizon_vs_{primary_metric}.{extension}'
                horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
            plt.close(horizon_figure)

        dataset_dir = ensure_directory(aggregate_dir / 'datasets')
        aggregate_metrics = metrics_frame[metrics_frame['horizon_index'].isna()].copy()
        for dataset_name in sorted(aggregate_metrics['dataset_name'].dropna().unique()):
            dataset_metrics = aggregate_metrics[aggregate_metrics['dataset_name'] == dataset_name].copy()
            if dataset_metrics.empty:
                continue
            dataset_slug = _slugify_model_name(str(dataset_name))

            for metric_name in sorted(dataset_metrics['metric_name'].dropna().unique()):
                metric_frame = dataset_metrics[dataset_metrics['metric_name'] == metric_name].copy()
                if metric_frame.empty or metric_frame['model_name'].nunique() == 0:
                    continue

                distribution_figure, distribution_axis = plt.subplots(figsize=(10, 5))
                metric_frame.boxplot(column='metric_value', by='model_name', ax=distribution_axis)
                distribution_axis.set_title(f'{dataset_name}: {metric_name.upper()} Distribution')
                distribution_axis.set_xlabel('Model')
                distribution_axis.set_ylabel(metric_name.upper())
                distribution_axis.figure.suptitle('')
                distribution_axis.grid(alpha=0.2)
                for extension in ('png', 'svg'):
                    path = dataset_dir / f'{dataset_slug}_metric_distribution_{metric_name}.{extension}'
                    distribution_figure.savefig(path, dpi=200, bbox_inches='tight')
                    manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(distribution_figure)

                ranking_frame = (
                    metric_frame.groupby('model_name')['metric_value']
                    .mean()
                    .sort_values()
                    .reset_index()
                )
                ranking_figure, ranking_axis = plt.subplots(figsize=(10, 5))
                ranking_axis.bar(ranking_frame['model_name'], ranking_frame['metric_value'])
                ranking_axis.set_title(f'{dataset_name}: Mean {metric_name.upper()} by Model')
                ranking_axis.set_xlabel('Model')
                ranking_axis.set_ylabel(metric_name.upper())
                ranking_axis.tick_params(axis='x', rotation=25)
                ranking_axis.grid(alpha=0.2, axis='y')
                for extension in ('png', 'svg'):
                    path = dataset_dir / f'{dataset_slug}_model_ranking_{metric_name}.{extension}'
                    ranking_figure.savefig(path, dpi=200, bbox_inches='tight')
                    manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                plt.close(ranking_figure)

                horizon_frame = metrics_frame[
                    (metrics_frame['dataset_name'] == dataset_name)
                    & (metrics_frame['metric_name'] == metric_name)
                    & (metrics_frame['horizon_index'].notna())
                    ].copy()
                if not horizon_frame.empty:
                    horizon_summary = (
                        horizon_frame.groupby(['model_name', 'horizon_index'])['metric_value']
                        .mean()
                        .reset_index()
                    )
                    dataset_horizon_figure, dataset_horizon_axis = plt.subplots(figsize=(10, 5))
                    for model_name, group in horizon_summary.groupby('model_name'):
                        dataset_horizon_axis.plot(
                            group['horizon_index'],
                            group['metric_value'],
                            label=model_name,
                            linewidth=1.8,
                        )
                    dataset_horizon_axis.set_title(f'{dataset_name}: Horizon-wise {metric_name.upper()}')
                    dataset_horizon_axis.set_xlabel('Horizon')
                    dataset_horizon_axis.set_ylabel(metric_name.upper())
                    dataset_horizon_axis.legend(frameon=False)
                    dataset_horizon_axis.grid(alpha=0.2)
                    for extension in ('png', 'svg'):
                        path = dataset_dir / f'{dataset_slug}_horizon_distribution_{metric_name}.{extension}'
                        dataset_horizon_figure.savefig(path, dpi=200, bbox_inches='tight')
                        manifest.append(ArtifactRecord(kind='plot', path=str(path), format=extension))
                    plt.close(dataset_horizon_figure)

        okhs_rows = successful_runs[successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
        if not okhs_rows.empty:
            baseline_rows = successful_runs[
                ~successful_runs['model_name'].str.contains('okhs', case=False, regex=False)]
            pairwise_rows = []
            for _, okhs_row in okhs_rows.iterrows():
                comparable = baseline_rows[
                    (baseline_rows['benchmark'] == okhs_row['benchmark'])
                    & (baseline_rows['dataset_name'] == okhs_row['dataset_name'])
                    & (baseline_rows['series_id'] == okhs_row['series_id'])
                    ]
                for _, baseline_row in comparable.iterrows():
                    pairwise_rows.append(
                        {
                            'benchmark': okhs_row['benchmark'],
                            'dataset_name': okhs_row['dataset_name'],
                            'series_id': okhs_row['series_id'],
                            'okhs_model': okhs_row['model_name'],
                            'baseline_model': baseline_row['model_name'],
                            primary_metric: okhs_row[primary_metric],
                            f'baseline_{primary_metric}': baseline_row[primary_metric],
                            'delta': okhs_row[primary_metric] - baseline_row[primary_metric],
                        }
                    )
            if pairwise_rows:
                pairwise_frame = pd.DataFrame(pairwise_rows).sort_values('delta')
                manifest.extend(_stable_write_table(pairwise_frame, aggregate_dir / 'okhs_pairwise_comparison'))

    available_series = predictions_frame['series_id'].drop_duplicates().tolist()
    for series_id in available_series[: min(3, len(available_series))]:
        comparison = compare_models_on_series(result, series_id=series_id, output_dir=series_dir / series_id)
        manifest.extend(comparison.artifact_manifest)

    return tuple(manifest)
