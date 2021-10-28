from typing import List, Optional

import numpy as np

from fedot.core.repository.tasks import TaskTypesEnum


class MultiModalData(dict):

    def __init__(self, *arg, **kw):
        super(MultiModalData, self).__init__(*arg, **kw)

    @property
    def idx(self):
        return next(iter(self.values())).idx

    @property
    def task(self):
        return next(iter(self.values())).task

    @property
    def target(self):
        return next(iter(self.values())).target

    @target.setter
    def target(self, value):
        for data_part in self.values():
            data_part.target = value

    @property
    def data_type(self):
        return [i.data_type for i in iter(self.values())]

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None

    def shuffle(self):
        # TODO implement multi-modal shuffle
        pass

    def subset_range(self, start: int, end: int):
        for key in self.keys():
            self[key] = self[key].subset_range(start, end)
        return self

    def subset_indices(self, selected_idx: List):
        for key in self.keys():
            self[key] = self[key].subset_indices(selected_idx)
        return self
