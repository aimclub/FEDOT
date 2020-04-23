from typing import (Any, List)


def node_height(chain: Any, node: Any) -> int:
    def recursive_child_height(parent_node: Any) -> int:
        node_child = chain.node_childs(parent_node)
        if node_child:
            height = recursive_child_height(node_child[0]) + 1
            return height
        else:
            return 0

    height = recursive_child_height(node)
    return height


def node_depth(node: Any) -> int:
    if not node.nodes_from:
        return 0
    else:
        return 1 + max([node_depth(next_node) for next_node in node.nodes_from])


def nodes_from_height(chain: Any, selected_height: int) -> List[Any]:
    def get_nodes(node: Any, current_height):
        nodes = []
        if current_height == selected_height:
            nodes.append(node)
        else:
            if node.nodes_from:
                for child in node.nodes_from:
                    nodes += get_nodes(child, current_height + 1)
        return nodes

    nodes = get_nodes(chain.root_node, current_height=0)
    return nodes
