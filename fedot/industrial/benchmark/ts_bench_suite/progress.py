from __future__ import annotations

from dataclasses import dataclass, field

try:  # pragma: no cover - dependency is expected, but keep a safe fallback
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    from tqdm import tqdm  # type: ignore

TQDM_FACTORY = tqdm
TQDM_WRITE = tqdm.write


@dataclass
class BenchmarkProgressMonitor:
    enabled: bool
    task_type: str
    run_name: str
    leave: bool = False
    log_errors: bool = True
    log_summaries: bool = True
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    not_available_count: int = 0
    current_dataset: str = ''
    current_model: str = ''
    current_item: str = ''
    current_dataset_counts: dict[str, int] = field(init=False, default_factory=dict)
    current_model_counts: dict[str, int] = field(init=False, default_factory=dict)
    bar: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.enabled:
            self.bar = TQDM_FACTORY(
                total=0,
                desc=f'{self.task_type}:{self.run_name}',
                unit='eval',
                dynamic_ncols=True,
                leave=self.leave,
            )

    def extend_total(self, amount: int) -> None:
        if not self.enabled or self.bar is None or amount <= 0:
            return
        total = getattr(self.bar, 'total', 0) or 0
        self.bar.total = total + amount
        if hasattr(self.bar, 'refresh'):
            self.bar.refresh()

    def dataset_loaded(self, dataset_name: str, item_count: int) -> None:
        self.current_dataset = dataset_name
        self.current_model = ''
        self.current_item = ''
        self.current_dataset_counts = self._empty_counts()
        self._write(f'[{self.task_type}] dataset={dataset_name} loaded items={item_count}')
        self._refresh(status='dataset_loaded')

    def model_started(self, dataset_name: str, model_name: str) -> None:
        self.current_dataset = dataset_name
        self.current_model = model_name
        self.current_item = ''
        self.current_model_counts = self._empty_counts()
        self._refresh(status='model_started')

    def item_started(self, dataset_name: str, model_name: str, item_name: str) -> None:
        self.current_dataset = dataset_name
        self.current_model = model_name
        self.current_item = item_name
        self._refresh(status='running')

    def advance(self, status: str, message: str = '') -> None:
        normalized = str(status)
        if normalized == 'success':
            self.success_count += 1
        elif normalized == 'failed':
            self.failed_count += 1
        elif normalized == 'skipped':
            self.skipped_count += 1
        elif normalized == 'not_available':
            self.not_available_count += 1

        self.current_dataset_counts[normalized] = self.current_dataset_counts.get(normalized, 0) + 1
        self.current_model_counts[normalized] = self.current_model_counts.get(normalized, 0) + 1

        if self.enabled and self.bar is not None and hasattr(self.bar, 'update'):
            self.bar.update(1)
        if message and self.log_errors and normalized != 'success':
            self._write(
                f'[{self.task_type}] status={normalized} dataset={self.current_dataset} '
                f'model={self.current_model} item={self.current_item or "-"} message={message}'
            )
        self._refresh(status=normalized)

    def model_finished(self) -> None:
        if not self.log_summaries or not self.current_model:
            return
        self._write(
            f'[{self.task_type}] model_summary dataset={self.current_dataset} '
            f'model={self.current_model} {self._format_counts(self.current_model_counts)}'
        )

    def dataset_finished(self) -> None:
        if not self.log_summaries or not self.current_dataset:
            return
        self._write(
            f'[{self.task_type}] dataset_summary dataset={self.current_dataset} '
            f'{self._format_counts(self.current_dataset_counts)}'
        )

    def close(self) -> None:
        if self.enabled and self.bar is not None and hasattr(self.bar, 'close'):
            self.bar.close()

    def _refresh(self, *, status: str) -> None:
        if not self.enabled or self.bar is None:
            return
        postfix = {
            'dataset': self.current_dataset[:24],
            'model': self.current_model[:24],
            'item': self.current_item[:18] if self.current_item else '-',
            'status': status,
            'ok': self.success_count,
            'fail': self.failed_count,
            'skip': self.skipped_count,
            'na': self.not_available_count,
        }
        if hasattr(self.bar, 'set_postfix'):
            self.bar.set_postfix(postfix, refresh=False)
        elif hasattr(self.bar, 'set_postfix_str'):
            self.bar.set_postfix_str(str(postfix))

    def _write(self, message: str) -> None:
        if self.enabled:
            TQDM_WRITE(message)

    @staticmethod
    def _empty_counts() -> dict[str, int]:
        return {'success': 0, 'failed': 0, 'skipped': 0, 'not_available': 0}

    @staticmethod
    def _format_counts(counts: dict[str, int]) -> str:
        return ' '.join(
            [
                f"ok={counts.get('success', 0)}",
                f"fail={counts.get('failed', 0)}",
                f"skip={counts.get('skipped', 0)}",
                f"na={counts.get('not_available', 0)}",
            ]
        )
