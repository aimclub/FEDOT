from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

gr = GraphRequirements()
gap = GPAlgorithmParameters()
ggp = GraphGenerationParams()
m = Mutation(parameters=gap, requirements=gr, graph_generation_params=ggp)

# model - a fitted ts-forecasting model
# take some individual
ind = model.history.individuals[-2][0]

# mutation
x = m._mutation(ind)
x
