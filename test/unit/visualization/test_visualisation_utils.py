from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_convert import chain_template_as_nx_graph
from fedot.core.chains.chain_template import ChainTemplate
from fedot.core.chains.node import SecondaryNode, PrimaryNode
from fedot.core.composer.visualisation import hierarchy_pos


def chain_first():  # tested chain
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def test_chain_template_as_nx_graph():
    chain = chain_first()
    chain_template = ChainTemplate(chain)
    graph, node_labels = chain_template_as_nx_graph(chain=chain_template)

    assert len(graph.nodes) == len(chain.nodes)  # check node quantity
    assert node_labels[0] == str(chain.root_node)  # check root node


def make_comparable_lists(pos, real_hierarchy_levels, node_labels, dim, reverse):
    def extract_levels(hierarchy_levels):
        levels = []
        for pair in hierarchy_levels:
            levels.append(sorted(pair[1]))
        return levels

    computed_hierarchy_levels = {}
    for node in pos:
        level = pos[node][dim]
        if level in computed_hierarchy_levels:
            computed_hierarchy_levels[level].append(node_labels[node])
        else:
            computed_hierarchy_levels[level] = [node_labels[node]]

    sorted_computed_hierarchy_levels = sorted(computed_hierarchy_levels.items(),
                                              key=lambda x: x[0], reverse=reverse)
    sorted_real_hierarchy_levels = sorted(real_hierarchy_levels.items(),
                                          key=lambda x: x[0])
    return extract_levels(sorted_computed_hierarchy_levels), \
           extract_levels(sorted_real_hierarchy_levels)


def test_hierarchy_pos():
    chain = chain_first()
    real_hierarchy_levels_y = {0: ['xgboost'], 1: ['xgboost', 'knn'],
                               2: ['logit', 'lda', 'logit', 'lda']}
    real_hierarchy_levels_x = {0: ['logit'], 1: ['xgboost'], 2: ['lda'],
                               3: ['xgboost'], 4: ['logit'], 5: ['knn'], 6: ['lda']}
    chain_template = ChainTemplate(chain)
    graph, node_labels = chain_template_as_nx_graph(chain=chain_template)
    pos = hierarchy_pos(graph.to_undirected(), root=0)
    comparable_lists_y = make_comparable_lists(pos, real_hierarchy_levels_y,
                                               node_labels, 1, reverse=True)
    comparable_lists_x = make_comparable_lists(pos, real_hierarchy_levels_x,
                                               node_labels, 0, reverse=False)
    assert comparable_lists_y[0] == comparable_lists_y[1]  # check nodes hierarchy by y axis
    assert comparable_lists_x[0] == comparable_lists_x[1]  # check nodes hierarchy by x axis
