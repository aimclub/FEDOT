from typing import Optional, Union, Tuple
from inspect import signature
from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn import tree

from fedot.explainability.explainer import Explainer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.composer.metrics import Metric, F1
from fedot.core.data.data import InputData


def _build_naive_surrogate_model(black_box_model, surrogate_model, data: InputData, metric: Metric = None, **kwargs) -> Optional[float]:
    
    output_mode = 'default'
    if data.task.task_type == TaskTypesEnum.classification:
        output_mode = 'labels'
        metric = F1 if metric is None else metric

    prediction = black_box_model.predict(data, output_mode=output_mode)
    surrogate_model.fit(data, prediction)

    if metric:
        data_c = deepcopy(data)
        data_c.target = surrogate_model.predict(data, output_mode=output_mode).predict
        score = -metric.metric(data_c, prediction)

    return score


class SurrogateExplainer(Explainer):
    surrogates_default_params = {
        'dt': {'max_depth': 3},
    }

    def __init__(self, model, surrogate: Union[str, Pipeline]):
        super().__init__(model)

        self.score: Optional[float] = None

        if isinstance(surrogate, Pipeline):
            self.surrogate = surrogate

        elif isinstance(surrogate, str):
            
            self.surrogate_str = surrogate
            
            if surrogate in self.surrogates_default_params:
                surrogate_node = PrimaryNode(surrogate)
                surrogate_node.custom_params = self.surrogates_default_params[surrogate]
                self.surrogate = Pipeline(surrogate_node)

            else:
                raise ValueError(f'Surrogate {surrogate} is not supported as surrogate model')
        else:
            raise ValueError(f'{type(surrogate)} is not supported as Fedot model')

    def __call__(self, data: InputData, plot: bool = True, **kwargs):
        try:
            self.score = _build_naive_surrogate_model(self.model, self.surrogate, data)
        
        except Exception as ex:
            print(f'Failed to fit the surrogate: {ex}')
            return

        if plot:
            self.plot(**kwargs)

    def plot(self, figsize: Tuple[int, int] = (16, 8), max_depth: int = None, proportion: bool = True, filled: bool = True, rounded: bool = True, **kwargs):
        
        plt.figure(figsize=figsize, dpi=100)
        if self.surrogate_str in ['dt']:

            if self.score is not None:
                print(f'Surrogate\'s model reproduction quality: {self.score}')
           
            # Tree parameters defined by user 
            kwargs_params = \
                {arg: kwargs[arg] for arg in kwargs if arg in signature(tree.plot_tree).parameters}


            tree.plot_tree(self.surrogate.root_node.fitted_operation,
                           max_depth=max_depth, proportion=proportion, filled=filled, rounded=rounded,
                           **kwargs_params)
