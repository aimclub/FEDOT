Main Concepts
=============

The main framework concepts are as follows:

- **Flexibility.** FEDOT can be used to automate the construction of solutions for various `problems <https://fedot.readthedocs.io/en/master/introduction/fedot_features/main_features.html#involved-tasks>`_, `data types <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#data-nature>`_ (texts, images, tables), and :doc:`models </advanced/automated_pipelines_design>`;
- **Extensibility.** Pipeline optimization algorithms are data- and task-independent, yet you can use :doc:`special strategies </api/strategies>` for specific tasks or data types (time-series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- **Integrability.** FEDOT supports widely used ML libraries (Scikit-learn, CatBoost, XGBoost, etc.) and allows you to integrate `custom ones <https://fedot.readthedocs.io/en/master/api/strategies.html#module-fedot.core.operations.evaluation.custom>`_;
- **Tuningability.** Various :doc:`hyper-parameters tuning methods </advanced/hyperparameters_tuning>` are supported including models' custom evaluation metrics and search spaces;
- **Versatility.** FEDOT is :doc:`not limited to specific modeling tasks </advanced/architecture>`, for example, it can be used in ODE or PDE;
- **Reproducibility.** Resulting pipelines can be :doc:`exported separately as JSON </advanced/pipeline_import_export>` or :doc:`together with your input data as ZIP archive </advanced/project_import_export>` for experiments reproducibility;
- **Customizability.** FEDOT allows `managing models complexity <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#models-used>`_ and thereby achieving desired quality.

The comparison of fedot with main existing AutoML tools is provided below:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow">Framework</th>
    <th class="tg-c3ow">Supported<br>pipeline<br>structure</th>
    <th class="tg-0pky">Models</th>
    <th class="tg-c3ow">Data types</th>
    <th class="tg-c3ow">Task types</th>
    <th class="tg-c3ow">Optimisation <br>algorithm</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">FEDOT</td>
    <td class="tg-c3ow">DAG</td>
    <td class="tg-0pky">Classical ML models, NNs, <br>self-implemented algorthims+ <br>support for custom models</td>
    <td class="tg-c3ow">Tabular, time series, <br>texts, multi-modal</td>
    <td class="tg-c3ow">Classification, regression, <br>forecasting + combinations</td>
    <td class="tg-c3ow">Adaptive evolution + <br>Optuna + support <br>for custom optimizers </td>
  </tr>
  <tr>
    <td class="tg-c3ow">AutoGluon</td>
    <td class="tg-c3ow">Ensemble</td>
    <td class="tg-0pky">Classical ML models + NNs</td>
    <td class="tg-c3ow">Tabular, time series, <br>texts, multi-modal</td>
    <td class="tg-c3ow">Classification, regression, <br>forecasting</td>
    <td class="tg-c3ow">Grid Search + Bayesian + <br>Optuna</td>
  </tr>
  <tr>
    <td class="tg-c3ow">H2O</td>
    <td class="tg-c3ow">Ensemble</td>
    <td class="tg-0pky">Classical ML models</td>
    <td class="tg-c3ow">Tabular, Texts</td>
    <td class="tg-c3ow">Classification, regression</td>
    <td class="tg-c3ow">Random Grid Search</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TPOT</td>
    <td class="tg-c3ow">Ensemble</td>
    <td class="tg-0pky">Classical ML models</td>
    <td class="tg-c3ow">Tabular</td>
    <td class="tg-c3ow">Classification, regression</td>
    <td class="tg-c3ow">GP</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LightAutoML</td>
    <td class="tg-c3ow">Ensemble</td>
    <td class="tg-0pky">Classical ML models + NNs</td>
    <td class="tg-c3ow">Tabular, time series, texts</td>
    <td class="tg-c3ow">Classification, regression, <br>forecasting</td>
    <td class="tg-c3ow">Heuristic + Optuna</td>
  </tr>
</tbody></table>                                            |        Tabular, time series, texts        |         Classification, regression,  forecasting        |                       Heuristic + Optuna                       |