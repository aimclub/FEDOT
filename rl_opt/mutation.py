from enum import Enum, auto


class PolicyMutationEnum(Enum):
    AddAction = auto()
    RemoveAction = auto()
    AddState = auto()
    RemoveState = auto()
    MergeStates = auto()
