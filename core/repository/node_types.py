from enum import Enum


class NodeType(Enum):
    pass


class SecondaryNodeType(NodeType):
    terminal = 'terminal'
    non_termianl = 'non_terminal'
