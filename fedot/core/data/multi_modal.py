from typing import Optional

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

    @property
    def data_type(self):
        return [i.data_type for i in iter(self.values())]

    @property
    def num_classes(self) -> Optional[int]:
        if self.task.task_type == TaskTypesEnum.classification:
            return len(np.unique(self.target))
        else:
            return None
