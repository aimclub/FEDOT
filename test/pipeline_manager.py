from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def default_valid_pipeline():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[first])
    final = SecondaryNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def baseline_pipeline():
    pipeline = Pipeline()
    last_node = SecondaryNode(operation_type='xgboost',
                              nodes_from=[])
    for requirement_model in ['knn', 'logit']:
        new_node = PrimaryNode(requirement_model)
        pipeline.add_node(new_node)
        last_node.nodes_from.append(new_node)
    pipeline.add_node(last_node)

    return pipeline


def pipeline_first():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    pipeline = Pipeline()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            pipeline.add_node(new_node)
        pipeline.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    pipeline.add_node(root_of_tree)
    return pipeline


def pipeline_second():
    #      XG
    #   |      \
    #  XG      KNN
    #  | \      |  \
    # LR XG   LR   LDA
    #    |  \
    #   KNN  LDA
    new_node = SecondaryNode('xgboost')
    for model_type in ('knn', 'lda'):
        new_node.nodes_from.append(PrimaryNode(model_type))
    pipeline = pipeline_first()
    pipeline.update_subtree(pipeline.root_node.nodes_from[0].nodes_from[1], new_node)

    return pipeline


def pipeline_third():
    #      XG
    #   /  |  \
    #  KNN LDA KNN
    root_of_tree = SecondaryNode('xgboost')
    for model_type in ('knn', 'lda', 'knn'):
        root_of_tree.nodes_from.append(PrimaryNode(model_type))
    pipeline = Pipeline()

    for node in root_of_tree.nodes_from:
        pipeline.add_node(node)
    pipeline.add_node(root_of_tree)

    return pipeline


def pipeline_fourth():
    #      XG
    #   |  \  \
    #  KNN  XG  KNN
    #      |  \
    #    KNN   KNN

    pipeline = pipeline_third()
    new_node = SecondaryNode('xgboost')
    [new_node.nodes_from.append(PrimaryNode('knn')) for _ in range(2)]
    pipeline.update_subtree(pipeline.root_node.nodes_from[1], new_node)

    return pipeline
