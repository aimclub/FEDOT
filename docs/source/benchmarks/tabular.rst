Tabular data
------------

Here are overall classification problem results across state-of-the-art AutoML frameworks
using self-runned tasks form OpenML test suite (10 folds run) using F1:


.. csv-table::
   :header: Dataset,FEDOT,AutoGluon,H2O,TPOT

    adult,0.874,0.874,0.875,0.874
    airlines,0.669,0.669,0.675,0.617
    airlinescodrnaadult,0.812,-,0.818,0.809
    albert,0.670,0.669,0.697,0.667
    amazon_employee_access,0.949,0.947,0.951,0.953
    apsfailure,0.994,0.994,0.995,0.995
    australian,0.871,0.870,0.865,0.860
    bank-marketing,0.910,0.910,0.910,0.899
    blood-transfusion,0.747,0.697,0.797,0.746
    car,1.000,1.000,0.998,0.998
    christine,0.746,0.746,0.748,0.737
    click_prediction_small,0.835,0.835,0.777,0.777
    cnae-9,0.957,0.954,0.957,0.954
    connect-4,0.792,0.788,0.865,0.867
    covertype,0.964,0.966,0.976,0.952
    credit-g,0.753,0.759,0.766,0.727
    dilbert,0.985,0.982,0.996,0.984
    fabert,0.688,0.685,0.726,0.534
    fashion-mnist,0.885,-,0.734,0.718
    guillermo,0.821,-,0.915,0.897
    helena,0.332,0.333,-,0.318
    higgs,0.731,0.732,0.369,0.336
    jannis,0.718,0.718,0.743,0.719
    jasmine,0.817,0.821,0.734,0.727
    jungle_chess_2pcs_raw_endgame_complete,0.953,0.939,0.817,0.817
    kc1,0.866,0.867,0.996,0.947
    kddcup09_appetency,0.982,0.982,0.866,0.818
    kr-vs-kp,0.995,0.996,0.982,0.962
    mfeat-factors,0.980,0.979,0.980,0.980
    miniboone,0.948,0.948,0.952,0.949
    nomao,0.969,0.970,0.975,0.974
    numerai28_6,0.523,0.522,0.522,0.505
    phoneme,0.915,0.916,0.916,0.910
    riccardo,0.997,-,0.998,0.997
    robert,0.405,-,0.559,0.487
    segment,0.982,0.982,0.982,0.980
    shuttle,1.000,1.000,1.000,1.000
    sylvine,0.952,0.951,0.952,0.948
    vehicle,0.851,0.849,0.846,0.835
    volkert,0.694,0.694,0.758,0.697
    Mean F1,0.838,0.837,0.833,0.812


Also, we tested FEDOT on the results of `AMLB <https://github.com/openml/automlbenchmark>`_ benchmark.
The visualization of FEDOT (v.0.7.3) results against H2O (3.46.0.4), AutoGluon (v.1.1.0), TPOT (v.0.12.1) and LightAutoML (v.0.3.7.3)
obtained using built-in visualizations of critial difference plot (CD) from AutoMLBenchmark [1].

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
obtained using sever based on Xeon Cascadelake (2900MHz) with 12 cores and 24GB memory.

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

[1] Gijsbers P. et al. AMLB: an AutoML benchmark //Journal of Machine Learning Research. – 2024. – Т. 25. – №. 101. – С. 1-65.