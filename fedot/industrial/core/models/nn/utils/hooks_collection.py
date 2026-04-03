from itertools import chain
from typing import Iterable, Optional

from fedot.industrial.core.models.nn.utils.hooks import BaseHook


class HooksCollection:
    def __init__(self, hooks: Optional[Iterable[BaseHook]] = None):
        self._on_epoch_start = []
        self._on_epoch_end = []
        for hook in hooks or ():
            self.append(hook)

    @property
    def start(self) -> list[BaseHook]:
        return self._on_epoch_start

    @property
    def end(self) -> list[BaseHook]:
        return self._on_epoch_end

    def all_hooks(self) -> list[BaseHook]:
        return self._on_epoch_start + self._on_epoch_end

    @staticmethod
    def _resolve_hook_place(hook: BaseHook) -> int:
        return getattr(hook, 'HOOK_PLACE', getattr(hook, '_hook_place', 0))

    def _sort_start(self):
        self._on_epoch_start.sort(key=self._resolve_hook_place)

    def _sort_end(self):
        self._on_epoch_end.sort(key=self._resolve_hook_place)

    def append(self, hook: BaseHook):
        if not isinstance(hook, BaseHook):
            raise TypeError(f'Expected BaseHook, got {type(hook).__name__}')

        hook_place = self._resolve_hook_place(hook)
        if hook_place > 0:
            self._on_epoch_end.append(hook)
            self._sort_end()
        elif hook_place < 0:
            self._on_epoch_start.append(hook)
            self._sort_start()
        else:
            self._on_epoch_start.append(hook)
            self._on_epoch_end.append(hook)
            self._sort_end()
            self._sort_start()

    def extend(self, hooks: Iterable[BaseHook]):
        for hook in hooks:
            self.append(hook)

    def clear(self):
        self._on_epoch_end.clear()
        self._on_epoch_start.clear()

    def check(self, additional_hooks):
        return self._check_specific(additional_hooks)
        # and other checks

    def _check_specific(self, hooks):
        if not hooks:
            return False
        iterable_hooks = chain(*hooks)
        hook_classes = tuple(hook.__class__ for hook in self.all_hooks())
        for specific_hook in iterable_hooks:
            if specific_hook.value in hook_classes:
                return True
        return False

    def __repr__(self):
        return "Training Scheme:\nEpoch start:\n\t{}\n<<<Training>>>\nEpoch end\n\t{}".format(
            '\n\t'.join(str(hook) for hook in self._on_epoch_start),
            '\n\t'.join(str(hook) for hook in self._on_epoch_end))
