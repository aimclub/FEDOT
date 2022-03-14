from typing import List, Union

from fedot.api.api_utils.assumptions_builder import AssumptionsBuilder
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.log import Log


class ApiInitialAssumptions:
    def get_initial_assumption(self,
                               data: Union[InputData, MultiModalData],
                               task: Task,
                               available_operations: List[str] = None,
                               logger: Log = None) -> List[Pipeline]:
        return AssumptionsBuilder\
            .get(task, data)\
            .with_logger(logger)\
            .from_operations(available_operations or [])\
            .build()
