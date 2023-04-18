Involved tasks
--------------

FEDOT is currently capable of solving:

* `Classification <https://github.com/stockblog/webinar_automl_fedot/blob/main/FEDOT%20Tutorial%20-%20Classification.ipynb>`_
* `Regression <https://github.com/stockblog/webinar_automl_fedot/blob/main/FEDOT%20Tutorial%20-%20Regression.ipynb>`_
* `Time-series forecasting ([uni/multi]variate) <https://github.com/stockblog/webinar_automl_fedot/blob/main/FEDOT%20Tutorial%20-%20Timeseries%20Forecasting.ipynb>`_


Pipeline building
-----------------

FEDOT uses open-source library named `GOLEM <https://github.com/aimclub/GOLEM#graph-optimization-and-learning-by-evolutionary-methods>`_
for optimization and learning of graph-based pipelines with meta-heuristic methods.

The library is potentially applicable to any graph-based optimization problem with clearly defined fitness function on it.

Sure enough, you may use your own custom optimization algorithms, see :doc:`/advanced/automated_pipelines_design`.
