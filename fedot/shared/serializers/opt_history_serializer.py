from typing import Any, Dict

from ..interfaces.serializable import Serializable


class OptHistorySerializer(Serializable):

    def to_json(self) -> Dict[str, Any]:
        circular_item = 'archive_history'
        serialized_obj = super().to_json()
        if circular_item in serialized_obj:
            for opt_hist_lst in serialized_obj[circular_item]:
                for opt_hist in opt_hist_lst:
                    if circular_item in opt_hist:
                        delattr(opt_hist, circular_item)
        return serialized_obj

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]):
        return super().from_json(json_obj)
