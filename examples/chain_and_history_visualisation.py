from fedot.core.chains.chain import Chain
from fedot.core.chains.node import SecondaryNode, PrimaryNode
from fedot.core.composer.composing_history import ComposingHistory
from fedot.core.composer.visualisation import ChainVisualiser


def chain_first():
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


def generate_history(generations, pop_size):
    history = ComposingHistory()
    for gen in range(generations):
        new_pop = []
        for idx in range(pop_size):
            chain = chain_first()
            chain.fitness = 1 / (gen * idx + 1)
            new_pop.append(chain)
        history.add_to_history(new_pop)
    return history


def run_chain_ang_history_visualisation(generations=2, pop_size=10,
                                        with_chain_visualisation=True):
    """ Function run visualisation of composing history and chain """
    # Generate chain and history
    chain = chain_first()
    history = generate_history(generations, pop_size)

    visualiser = ChainVisualiser()
    visualiser.visualise_history(history)
    if with_chain_visualisation:
        visualiser.visualise(chain)


if __name__ == '__main__':
    run_chain_ang_history_visualisation()
