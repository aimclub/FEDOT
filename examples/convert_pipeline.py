from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from test.unit.tasks.test_regression import get_synthetic_regression_data


# TODO remove this example - into test
node_scaling = PrimaryNode('scaling')
node_norm = PrimaryNode('normalization')
node_dtreg = SecondaryNode('dtreg', nodes_from=[node_scaling])
node_lasso = SecondaryNode('lasso', nodes_from=[node_norm])
node_final = SecondaryNode('ridge', nodes_from=[node_dtreg, node_lasso])
node_final.custom_params = {'alpha': 12.1}
pipeline = Pipeline(node_final)

# Generate data
input_data = get_synthetic_regression_data(n_samples=10, n_features=2,
                                           random_state=2021)
# Init fit
pipeline.fit(input_data)
init_preds = pipeline.predict(input_data)

# Convert into OptGraph object
adapter = PipelineAdapter()
opt_graph = adapter.adapt(pipeline)

restored_pipeline = adapter.restore(opt_graph)

pipeline.fit(input_data)
restored_preds = restored_pipeline.predict(input_data)

print(f'Init pipeline predictions: {init_preds.predict[:2]}')
print(f'Restored pipeline predictions: {restored_preds.predict[:2]}')
finish = True
