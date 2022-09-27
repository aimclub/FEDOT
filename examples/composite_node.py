
from fedot.core.optimisers.graph import OptNode


class CompositeNode(OptNode):
    def __str__(self):
        return self.content["name"]