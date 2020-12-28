from fedot.core.chains.chain import Chain
from fedot.core.chains.node import SecondaryNode, PrimaryNode
from fedot.core.composer.gp_opt_history import GPOptHistory
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


def generate_history(generations_quantity, pop_size):
    history = GPOptHistory()
    for gen in range(generations_quantity):
        new_pop = []
        for idx in range(pop_size):
            chain = chain_first()
            chain.fitness = 1 / (gen * idx + 1)
            new_pop.append(chain)
        history.add_to_history(new_pop)
    return history


if __name__ == '__main__':
    generations_quantity = 2
    pop_size = 10

    chain = chain_first()

    history = generate_history(generations_quantity, pop_size)
    history.prepare_for_visualisation()

    visualiser = ChainVisualiser()
    visualiser.visualise_history(history)
    visualiser.visualise(chain, params_save_path=r'C:\Users\user\Downloads\params.csv')
