Tabular data
------------

We tested FEDOT on the results of `AMLB <https://github.com/openml/automlbenchmark>`_ benchmark.
We used the setup of the framework obtained from 'frameworks.yaml' on the date of starts of experiments.
So, the following stable versions were used: AutoGluon 0.7.0, TPOT 0.11.7, LightAutoML 0.3.7.3, v3.40.0.2, FEDOT 0.7.2.
Some runs for AutoGluon are failed due to the errors (described also in Appendix D of AMLB paper [1]).

The visualization obtained using built-in visualizations of critical difference plot (CD) from AutoMLBenchmark [1].

In a CD (Critical Difference) diagram,
we display each framework's average rank and highlight which ranks are
statistically significantly different from one another.

To determine the average rank per task,
we first replace any missing values with a constant predictor,
calculate ranks for represented AutoML solutions and constant predictor
for each dataset and than took an average value of ranks across all datasets for each represented solution.

We assess statistical significance of the rank differences using a non-parametric Friedman test with a
threshold of p < 0.05 (resulting in p ≈ 0 for all diagrams)
and apply a Nemenyi post-hoc test to identify which framework pairs differ significantly.

Time budget for all experiments is 1 hour, 10 folds are used (1h8c setup for ALMB). The results are
obtained using sever based on Xeon Cascadelake (2900MHz) with 12 cores and 16GB memory.

CD for all datasets (ROC AUC and negative log loss):

.. image:: ./img_benchmarks/cd-all-1h8c-constantpredictor.png

The CD diagram for all datasets (ROC AUC and negative log loss) shows that all AutoML frameworks
(LightAutoML, H2OAutoML, TPOT,  AutoGluon, FEDOT) perform statistically better than constant predictor:

CD for binary classification (ROC AUC):

.. image:: ./img_benchmarks/cd-binary-classification-1h8c-constantpredictor.png

The CD diagram for binary classification (ROC AUC) shows that all AutoML frameworks
(LightAutoML, H2OAutoML, TPOT,  AutoGluon, FEDOT) perform similarly,
falling within the same CD interval, and significantly outperform  the constant predictor:

CD for multiclass classification (negative logloss):

.. image:: ./img_benchmarks/cd-multiclass-classification-1h8c-constantpredictor.png

The CD diagram for multiclass classification (negative log loss) shows that
TPOT and Fedot demonstrate intermediate performance being on the border of the
CD interval with constant predictor and the CD interval with H2OAutoML:

We can conclude that FEDOT achieves performance comparable with competitors for tabular tasks.

The ranks for frameworks are provided below:

.. image:: ./img_benchmarks/ranks.png

The raw metrics (ROC AUC for binary and logloss for multiclass) for frameworks are provided below:

.. image:: ./img_benchmarks/metrics.png

The comparison with [1] shows that AutoGluon is underperforming in our hardware setup,
while TPOT and H2O are quite close in both setups.
To avoid any confusion, we provide below an additional comparison of the FEDOT metrics with the metrics from [1].
However, it should be noted that the conditions are different, as are the exact versions of the frameworks.

.. image:: ./img_benchmarks/fedot_amlb.png

[1] Gijsbers P. et al. AMLB: an AutoML benchmark //Journal of Machine Learning Research. – 2024. – Т. 25. – №. 101. – С. 1-65.

