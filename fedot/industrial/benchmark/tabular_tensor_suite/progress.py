from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import BenchmarkStage

try:  # pragma: no cover - dependency is expected, but keep a safe fallback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write if tqdm is not None else None


@dataclass
class TabularBenchmarkProgressMonitor:
    enabled: bool
    dataset_total: int
    leave: bool = False
    log_errors: bool = True
    current_dataset: str = ''
    current_seed: str = ''
    current_mode: str = ''
    current_stage: str = ''
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    dataset_bar: Any | None = field(init=False, default=None)
    seed_bar: Any | None = field(init=False, default=None)
    mode_bar: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled and TQDM_FACTORY is not None)
        if not self.enabled:
            return
        self.dataset_bar = TQDM_FACTORY(
            total=self.dataset_total,
            desc='tabular-suite datasets',
            unit='dataset',
            position=0,
            dynamic_ncols=True,
            leave=self.leave,
        )

    def dataset_started(self, dataset_name: str, total_seeds: int) -> None:
        self.current_dataset = dataset_name
        self.current_seed = ''
        self.current_mode = ''
        self.current_stage = ''
        self._replace_bar(
            attribute_name='seed_bar',
            total=total_seeds,
            desc=f'{dataset_name} seeds',
            unit='seed',
            position=1,
        )
        self._replace_bar(
            attribute_name='mode_bar',
            total=0,
            desc=f'{dataset_name} modes',
            unit='mode',
            position=2,
        )
        self.write(f'[tabular_suite] dataset={dataset_name} stage={BenchmarkStage.LOAD.value}')
        self._refresh_stage()

    def dataset_finished(self) -> None:
        if self.enabled and self.dataset_bar is not None and hasattr(self.dataset_bar, 'update'):
            self.dataset_bar.update(1)
        self.current_seed = ''
        self.current_mode = ''
        self.current_stage = ''
        self._refresh_stage()

    def seed_started(self, dataset_name: str, seed: int, total_modes: int) -> None:
        self.current_dataset = dataset_name
        self.current_seed = str(seed)
        self.current_mode = ''
        self.current_stage = ''
        self._replace_bar(
            attribute_name='mode_bar',
            total=total_modes,
            desc=f'{dataset_name} seed={seed} modes',
            unit='mode',
            position=2,
        )
        self.write(f'[tabular_suite] dataset={dataset_name} seed={seed} stage={BenchmarkStage.SPLIT.value}')
        self._refresh_stage()

    def seed_finished(self) -> None:
        if self.enabled and self.seed_bar is not None and hasattr(self.seed_bar, 'update'):
            self.seed_bar.update(1)
        self.current_mode = ''
        self.current_stage = ''
        self._refresh_stage()

    def mode_started(self, mode: str) -> None:
        self.current_mode = mode
        self.current_stage = BenchmarkStage.LOAD.value
        self._refresh_stage()

    def stage_started(self, stage: BenchmarkStage, message: str = '') -> None:
        self.current_stage = stage.value
        if message:
            self.write(
                f'[tabular_suite] dataset={self.current_dataset} seed={self.current_seed or "-"} '
                f'mode={self.current_mode or "-"} stage={stage.value} {message}'
            )
        else:
            self.write(
                f'[tabular_suite] dataset={self.current_dataset} seed={self.current_seed or "-"} '
                f'mode={self.current_mode or "-"} stage={stage.value}'
            )
        self._refresh_stage()

    def mode_finished(self, status: str, message: str = '') -> None:
        normalized = str(status)
        if normalized == 'success':
            self.success_count += 1
        elif normalized == 'failed':
            self.failed_count += 1
        elif normalized == 'skipped':
            self.skipped_count += 1

        if self.enabled and self.mode_bar is not None and hasattr(self.mode_bar, 'update'):
            self.mode_bar.update(1)
        if message and self.log_errors and normalized != 'success':
            self.write(
                f'[tabular_suite] dataset={self.current_dataset} seed={self.current_seed or "-"} '
                f'mode={self.current_mode or "-"} status={normalized} message={message}'
            )
        self.current_stage = normalized
        self._refresh_stage()

    def close(self) -> None:
        for bar in (self.mode_bar, self.seed_bar, self.dataset_bar):
            if self.enabled and bar is not None and hasattr(bar, 'close'):
                bar.close()

    def write(self, message: str) -> None:
        if not self.enabled:
            return
        if TQDM_WRITE is not None:
            TQDM_WRITE(message)

    def _refresh_stage(self) -> None:
        if not self.enabled:
            return
        postfix = {
            'seed': self.current_seed or '-',
            'mode': self.current_mode or '-',
            'stage': self.current_stage or '-',
            'ok': self.success_count,
            'fail': self.failed_count,
            'skip': self.skipped_count,
        }
        for bar in (self.dataset_bar, self.seed_bar, self.mode_bar):
            if bar is None:
                continue
            if hasattr(bar, 'set_postfix'):
                bar.set_postfix(postfix, refresh=False)
            elif hasattr(bar, 'set_postfix_str'):
                bar.set_postfix_str(str(postfix))

    def _replace_bar(self, attribute_name: str, total: int, desc: str, unit: str, position: int) -> None:
        if not self.enabled:
            return
        current_bar = getattr(self, attribute_name)
        if current_bar is not None and hasattr(current_bar, 'close'):
            current_bar.close()
        setattr(
            self,
            attribute_name,
            TQDM_FACTORY(
                total=total,
                desc=desc,
                unit=unit,
                position=position,
                dynamic_ncols=True,
                leave=self.leave,
            ),
        )

