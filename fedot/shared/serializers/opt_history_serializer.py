from typing import Any, Dict

from fedot.core.optimisers.opt_history import OptHistory

from ..interfaces.serializable import Serializable


class OptHistorySerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        duplicate_fields = set([
            'historical_fitness', 'all_historical_fitness',
            'all_historical_quality', 'short_metrics_names',
            'historical_pipelines', 'parent_operators'
        ])
        return {
            k: v
            for k, v in super().to_json().items()
            if k not in duplicate_fields
        }

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        obj: OptHistory = super().from_json(json_obj)
        obj.parent_operators = [
            [ind.parent_operators for ind in ind_lst]
            for ind_lst in obj.individuals
        ]
        return obj
