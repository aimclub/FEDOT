# Gap report: tabular AutoML models and data operations for FEDOT

Дата: 2026-06-11.

Scope: только табличные задачи classification/regression. Time series, text/image/multimodal как самостоятельные задачи не рассматривались, но табличные операции для datetime/text-like columns отмечены, если они применяются внутри табличного AutoML.

## Baseline FEDOT

Проверенные источники в FEDOT:

- `fedot/core/repository/data/model_repository.json`
- `fedot/core/repository/data/data_operation_repository.json`
- `fedot/core/operations/evaluation/evaluation_interfaces.py`
- `fedot/core/operations/evaluation/common_preprocessing.py`
- `fedot/core/operations/evaluation/classification.py`
- `fedot/core/operations/evaluation/regression.py`

Табличные модели, которые уже есть:

- Boosting/tree ensembles: `catboost`, `catboostreg`, `lgbm`, `lgbmreg`, `xgboost`, `xgboostreg`, `gbr`, `adareg`, `rf`, `rfr`, `treg` (`ExtraTreesRegressor` only), `dt`, `dtreg`.
- Linear/SVM/kNN/NB/DA: `logit`, `linear`, `ridge`, `lasso`, `sgdr`, `svc`, `svr`, `knn`, `knnreg`, `bernb`, `multinb`, `lda`, `qda`.
- Neural/foundation: `mlp` (`MLPClassifier`), `tabpfn`, `tabpfnreg`, `tabicl`, `tabiclreg`, `cnn` (classification, not a standard tabular baseline).
- Other: `kmeans`, `custom`.

Табличные data operations, которые уже есть:

- Scaling/imputation/encoding: `scaling`, `normalization`, `simple_imputation`, `one_hot_encoding`, `label_encoding`.
- Dimension/features: `pca`, `dask_pca`, `kernel_pca`, `fast_ica`, `poly_features`.
- Selection/filtering/outliers: `rfe_lin_*`, `rfe_non_lin_*`, `ransac_*`, `isolation_forest_*`, `resample`, `decompose`, `class_decompose`.

Important gaps in this baseline:

- Нет `ExtraTreesClassifier`, хотя regression-вариант есть как `treg`.
- Нет `AdaBoostClassifier` и sklearn `GradientBoostingClassifier`, хотя regression-варианты `adareg`/`gbr` есть.
- Нет `ElasticNet`, `LassoLars`, `BayesianRidge`/`ARDRegression`, `GaussianProcessRegressor`, `PassiveAggressive`.
- Нет `GaussianNB`.
- Нет `HistGradientBoostingClassifier/Regressor`.
- Нет `DummyClassifier/Regressor` как explicit baseline operation.
- `MLPClassifier` есть, но нет `MLPRegressor` и нет современных tabular-DL моделей с категориальными embedding/BatchEnsemble/foundation-like inference.
- Препроцессинг покрывает PCA/KernelPCA/ICA/poly/OHE, но нет `TruncatedSVD`, `Nystroem`, `RBFSampler`, `RandomTreesEmbedding`, `FeatureAgglomeration`, `VarianceThreshold`, `SelectPercentile`, `SelectFwe`, `SelectFromModel`, `Binarizer`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`.
- Нет target/frequency/count encoding, datetime feature generation, missing-value indicator columns, rare-category handling/coalescing as standalone searchable operations.

## Framework scan

### AutoGluon Tabular 1.5.0

Checked locally in `.venv` and cross-checked with docs.

Key local files:

- `.venv/lib/python3.10/site-packages/autogluon/tabular/configs/hyperparameter_configs.py`
- `.venv/lib/python3.10/site-packages/autogluon/tabular/models/*`
- `.venv/lib/python3.10/site-packages/autogluon/features/generators/auto_ml_pipeline.py`

Relevant models/operations:

| AutoGluon item | FEDOT status | Importance | Notes for FEDOT |
|---|---:|---:|---|
| `XT` / ExtraTreesClassifier | Missing for classification | High | Very cheap ensemble baseline, often useful in stacked/ensemble systems. FEDOT already has `ExtraTreesRegressor` (`treg`), so classifier is a low-risk symmetry addition. |
| `NN_TORCH` | Partially missing | Medium | FEDOT has sklearn `MLPClassifier`, but not AutoGluon-style tabular torch NN with categorical embeddings, batch norm/dropout, early stopping, regression support. |
| `FASTAI` tabular NN | Missing | Medium | Similar role to `NN_TORCH`, but fastai tabular learner. Heavier dependency; probably not first choice unless FEDOT wants optional neural preset. |
| `REALMLP` | Missing | High | AutoGluon wraps `pytabkit` RealMLP. It is an improved MLP with strong defaults, missing indicators, bool-to-category handling, CPU/GPU support. Good modern neural baseline. |
| `TABM` | Missing | High | Efficient parameter-shared MLP ensemble. AutoGluon treats it as a strong TabArena-era tabular DL model. Good candidate if optional torch dependency is acceptable. |
| `TabPFNv2`, `TabPFNMix`, `TabDPT`, `Mitra` | Partially missing | Medium/High | FEDOT already has `TabPFN` and `TabICL`, but AutoGluon has newer variants and companion foundation models. Best handled as optional `non-default` operations with dataset-size/resource constraints. |
| `EBM` | Missing | Medium | Explainable Boosting Machine from InterpretML: additive/interpretable model with pairwise interactions. Useful for interpretable preset, not necessarily default. |
| `IM_RULEFIT`, `IM_FIGS`, `IM_GREEDYTREE`, `IM_BOOSTEDRULES`, `IM_HSTREE` | Missing | Medium | Interpretable rule/tree models via `imodels`. AutoGluon has an `interpretable` preset using RuleFit/FIGS. |
| `AutoMLPipelineFeatureGenerator` datetime/text special/ngram/image-path flags | Partially missing | Medium | For tabular data AutoGluon auto-generates datetime integer features, text statistics/ngrams, category memory optimization, NaN indicators for image-path fields. FEDOT has text ops, but not the same integrated tabular feature generator. |
| `CategoryFeatureGenerator`, `oof_target_encoder`, frequency/binned/drop-unique/drop-duplicates generators | Missing/partial | High for categorical ops | Target/frequency/count-like encoders and rare/drop operations are common high-value preprocessing additions for tabular tasks. |

Implementation differences:

- AutoGluon models are not just independent estimators; they are designed for bagging/stacking/weighted ensembles. FEDOT can add operations independently, but model payoff is larger if composition/tuning can discover them reliably.
- AutoGluon uses model-specific preprocessing. `RealMLP`/`TabM` do mean imputation plus missing indicators and special categorical handling inside the model wrapper. In FEDOT, this should either be inside the operation implementation or represented as explicit preprocessing nodes; avoid duplicating API-level normalization.
- AutoGluon `extreme_quality` in 1.5 uses TabPFNv2/TabICL/TabDPT/Mitra for datasets up to roughly 100k rows and requires GPU for the intended quality preset.

Sources:

- AutoGluon docs: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
- Local AutoGluon 1.5.0 code in `.venv`.

### FLAML 2.6.0

Installed into `.venv` and checked locally.

Key local files:

- `.venv/lib/python3.10/site-packages/flaml/automl/task/generic_task.py`
- `.venv/lib/python3.10/site-packages/flaml/automl/model.py`
- `.venv/lib/python3.10/site-packages/flaml/automl/contrib/histgb.py`

Relevant models/operations:

| FLAML item | FEDOT status | Importance | Notes for FEDOT |
|---|---:|---:|---|
| `extra_tree` | Missing for classification | High | Same recommendation as AutoGluon. FLAML includes ExtraTrees in default estimator list. |
| `xgb_limitdepth` | Missing as separate operation | Low/Medium | FEDOT has XGBoost. A separate limited-depth alias may be useful only if search space should strongly bias shallow XGBoost under small budgets. Could be a preset/search-space variant instead of new operation. |
| `lrl1`, `lrl2` | Partially present | Medium | FEDOT has `logit`, but not explicit L1/L2 variants as separate operations. Prefer hyperparameter space improvement over adding multiple operation names unless composition benefits from distinction. |
| `histgb` | Missing | High | scikit-learn Histogram Gradient Boosting supports native-ish missing handling and is strong without external LightGBM/XGBoost/CatBoost deps. Good fallback for pure sklearn environments. |
| `enet` | Missing | Medium | ElasticNet regression. Useful sparse/linear baseline between Lasso/Ridge. Low implementation risk. |
| `lassolars` | Missing | Low/Medium | Useful for high-dimensional linear regression; less generally important than ElasticNet. |
| `kneighbor` | Already present | - | FEDOT has `knn`/`knnreg`. FLAML drops categorical columns for kNN internally; FEDOT behavior should remain through existing preprocessing pipeline. |
| `transformer` | Out of scope mostly | Low for tabular-only | FLAML's transformer path is NLP, not a direct tabular model candidate here. |

Implementation differences:

- FLAML explicitly uses cost-aware search spaces. Several items are not new model families but constrained variants with low-cost initialization (`xgb_limitdepth`, small initial RF/ExtraTrees). For FEDOT this may belong in hyperparameter/search-space defaults, not operation repository expansion.
- FLAML `HistGradientBoostingEstimator` is in `contrib`, but has a clear sklearn implementation path.

Sources:

- FLAML local code in `.venv`.
- FLAML project/docs: https://microsoft.github.io/FLAML/

### auto-sklearn

Checked upstream docs and GitHub component directories.

Relevant models/operations:

| auto-sklearn item | FEDOT status | Importance | Notes for FEDOT |
|---|---:|---:|---|
| `extra_trees` classifier | Missing | High | Repeated across AutoGluon/FLAML/TPOT/auto-sklearn. Strong candidate. |
| `adaboost`, `gradient_boosting` classifiers | Missing/partial | Medium | FEDOT has `adareg`/`gbr` for regression and stronger external boosting classifiers, but sklearn classifier symmetry is missing. |
| `gaussian_nb` | Missing | Medium | Complements Bernoulli/Multinomial NB for continuous features. Very cheap classifier. |
| `passive_aggressive` | Missing | Low/Medium | Linear online-style classifier. Useful for large sparse data, but less central for dense tabular tasks. |
| `ard_regression` | Missing | Low/Medium | Bayesian sparse linear regression. Useful but niche. |
| `gaussian_process` regressor | Missing | Low | Expensive and scales poorly; maybe non-default only. |
| `liblinear_svc` vs `libsvm_svc` | Partially present | Low | FEDOT has `svc`. Separate linear/kernel operation split may help search cost control but can also be handled by params. |
| `densifier` | Missing | Low | Utility for sparse-to-dense compatibility; only needed if FEDOT expands sparse preprocessing. |
| `extra_trees_preproc_*` | Missing | Medium | Tree-based supervised feature selection/transform. Similar to `SelectFromModel(ExtraTrees*)`. |
| `feature_agglomeration` | Missing | Medium | Clusters similar features and replaces them with cluster aggregates. Good for high-dimensional correlated numeric features. |
| `kitchen_sinks` / `RBFSampler` | Missing | Medium | Random Fourier features for approximate RBF kernels. Useful before linear models. |
| `nystroem_sampler` | Missing | Medium | Kernel approximation. Useful alternative to `kernel_pca` with controllable feature count. |
| `random_trees_embedding` | Missing | Medium | Unsupervised tree embedding; can feed linear models/boosting. |
| `select_percentile`, `select_rates`, `truncatedSVD` | Missing | Medium/High | Common feature selection and sparse dimensionality reduction. `TruncatedSVD` is especially useful for sparse OHE/text-like tables. |

Implementation differences:

- auto-sklearn separates data preprocessing and feature preprocessing. Its docs explicitly distinguish mandatory data preparation (OHE, imputation, normalization) from searchable feature preprocessing such as PCA/selection.
- It uses ensemble selection over validation predictions. FEDOT has graph composition; adding models alone will not exactly reproduce auto-sklearn's ensemble behavior.

Sources:

- Manual: https://automl.github.io/auto-sklearn/master/manual.html
- Classifier components: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification
- Regressor components: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression
- Feature preprocessors: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/feature_preprocessing

### TPOT

Checked upstream configuration source.

Relevant models/operations:

| TPOT item | FEDOT status | Importance | Notes for FEDOT |
|---|---:|---:|---|
| `ExtraTreesClassifier` | Missing | High | Same repeated candidate. |
| `AdaBoostClassifier`, `GradientBoostingClassifier` | Missing/partial | Medium | FEDOT has regression variants and external boosting classifiers. Add only if sklearn-only preset completeness matters. |
| `GaussianNB` | Missing | Medium | Same as auto-sklearn. |
| `MLPRegressor` | Missing | Medium | FEDOT has `MLPClassifier`, not regressor. A sklearn MLP regressor is simple and fills symmetry. |
| `Binarizer` | Missing | Low/Medium | Thresholds numeric features; can help linear/NB models. |
| `FeatureAgglomeration` | Missing | Medium | Repeated with auto-sklearn. |
| `Nystroem`, `RBFSampler` | Missing | Medium | Repeated kernel approximation candidates. |
| `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `StandardScaler` split | Partially present | Medium | FEDOT has generic `scaling`/`normalization`, but explicit robust/minmax/maxabs variants may improve search behavior and sparse compatibility. |
| `ZeroCount` | Missing | Low/Medium | TPOT builtin that adds per-row counts of zero and non-zero values. Cheap meta-feature for sparse/binary tables. |
| `SelectFwe`, `SelectPercentile`, `VarianceThreshold`, `SelectFromModel` | Missing/partial | Medium/High | FEDOT has RFE, but not these cheaper selectors. `VarianceThreshold` is a very low-risk first addition. |

Implementation differences:

- TPOT treats many preprocessors/selectors as genetic-programming pipeline nodes. FEDOT can map them naturally to data operations, but should set tags carefully so selectors are not placed after incompatible nodes.
- TPOT's config includes OHE with `minimum_fraction` and category threshold behavior; FEDOT has OHE/label encoding but not rare-category frequency control.

Sources:

- TPOT classifier config: https://raw.githubusercontent.com/EpistasisLab/tpot/master/tpot/config/classifier.py
- TPOT regressor config: https://raw.githubusercontent.com/EpistasisLab/tpot/master/tpot/config/regressor.py

### H2O AutoML

Checked official H2O AutoML docs.

Relevant models/operations:

| H2O item | FEDOT status | Importance | Notes for FEDOT |
|---|---:|---:|---|
| XRT / Extremely Randomized Trees inside `DRF` | Missing for classification | High | Same family as ExtraTrees. H2O includes XRT under DRF. |
| `DeepLearning` fully-connected NN | Partially missing | Medium | FEDOT has `MLPClassifier`, not MLP regressor / modern tabular NN. H2O's NN is a generic dense neural baseline. |
| `GLM` with regularization | Partially present | Medium | FEDOT has linear/ridge/lasso/logit, but not full GLM family or ElasticNet-style unified regularization. |
| `StackedEnsemble` | Conceptually present via composition, not same | Medium | H2O builds explicit ensembles of all/base-family models using CV or blending. FEDOT composition can ensemble, but not with the same AutoML-level stack generation as a single operation. |
| `target_encoding` preprocessing | Missing | High | H2O exposes target encoding as experimental AutoML preprocessing. Very important for high-cardinality categorical data. |
| `monotone_constraints` for GBM/XGBoost | Mostly missing as operation-level tuning surface | Medium | Not a new model, but useful domain constraint support. Could be params rather than operation. |

Implementation differences:

- H2O is Java/backend based; model wrappers are not drop-in sklearn estimators. For FEDOT, prefer sklearn/native Python equivalents (`ExtraTrees*`, sklearn `HistGradientBoosting*`, optional torch models) rather than H2O dependency.
- H2O AutoML tightly couples CV predictions with stacked ensembles. Replicating this as a single operation would conflict with FEDOT graph semantics; better to improve ensemble/composition behavior separately.

Source:

- H2O AutoML docs: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

## Cross-framework candidates

### Highest priority

1. `ExtraTreesClassifier`
   - Appears in AutoGluon, FLAML, auto-sklearn, TPOT, H2O/XRT.
   - FEDOT already has `ExtraTreesRegressor` as `treg`; implementation should be small.
   - Suggested FEDOT operation name: `extra_trees` or `et`/`etc`. Avoid overloading `treg`.
   - Add to `model_repository.json`, sklearn classification strategy map, default params, and direct tests for repository availability and fit/predict.

2. `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`
   - Appears in FLAML contrib and is a strong pure-sklearn fallback.
   - Good when external boosting dependencies are unavailable or when missing-value support is useful.
   - Suggested operation names: `hist_gb`, `hist_gbreg`.

3. Target/frequency/count categorical encoders
   - AutoGluon has target/frequency-related generators; H2O has target encoding; TPOT has rare-frequency control in OHE.
   - FEDOT currently has OHE and label encoding only.
   - Suggested operations: `target_encoding`, `frequency_encoding`, maybe `rare_category_grouping`.
   - Implementation should avoid leakage: target encoding must be fold-aware for fit-time transform. This needs more than a plain sklearn transformer if used inside CV/composition.

4. Cheap selectors: `VarianceThreshold`, `SelectPercentile`, `SelectFwe`, `SelectFromModel`
   - Repeated in TPOT/auto-sklearn.
   - FEDOT has RFE, but these are faster and often enough.
   - Suggested operations: `variance_threshold`, `select_percentile_class/reg`, `select_fwe_class/reg`, `select_from_model_class/reg`.

5. `TruncatedSVD`
   - Important for sparse/high-dimensional OHE/text-like tabular matrices.
   - Complements PCA, which is not ideal for sparse input.
   - Suggested operation name: `truncated_svd`.

### High priority if neural/tabular-DL expansion is intended

1. `TabM`
   - Strong modern tabular DL candidate from AutoGluon 1.4/1.5.
   - Parameter-efficient ensemble of MLPs; better fit than generic deep nets for tabular data.
   - Should be optional/non-default at first, with CPU/GPU tags and dataset-size/resource rules.

2. `RealMLP`
   - Another strong modern MLP baseline from AutoGluon.
   - Needs careful model-specific preprocessing: numeric imputation, missing indicators, categorical handling.
   - Could be implemented via optional `pytabkit` dependency.

3. `MLPRegressor`
   - Simple symmetry addition if FEDOT keeps sklearn MLP classifier.
   - Lower performance upside than `TabM`/`RealMLP`, but cheap to add.

4. Newer foundation tabular models: `TabPFNv2`, `TabPFNMix`, `TabDPT`, `Mitra`
   - FEDOT already has `TabPFN` and `TabICL`, so this is an extension rather than a new family.
   - Best as `non-default` operations with strict dataset-size and dependency gates.

### Medium priority

1. `GaussianNB`
   - Cheap continuous-feature Naive Bayes classifier.
   - Complements existing `bernb` and `multinb`.

2. `AdaBoostClassifier` and sklearn `GradientBoostingClassifier`
   - Fill classifier/regressor symmetry for existing `adareg` and `gbr`.
   - Lower priority because FEDOT already has CatBoost/LightGBM/XGBoost classifiers.

3. `ElasticNet`
   - Useful regression baseline between Ridge and Lasso.
   - Easy sklearn implementation.

4. `LassoLars`, `ARDRegression`, `BayesianRidge`
   - Useful for some high-dimensional/sparse regression tasks.
   - Lower general payoff than ElasticNet.

5. Kernel approximations: `Nystroem`, `RBFSampler`
   - Useful before linear models as a cheaper alternative to exact kernel models.
   - Should be tagged as preprocessing and probably limited for large dense datasets.

6. `FeatureAgglomeration`, `RandomTreesEmbedding`
   - Useful high-dimensional transformations.
   - More risk of odd feature semantics; add after cheap selectors/SVD.

7. Scaling variants: `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`
   - FEDOT has generic `scaling` and `normalization`, but explicit operations improve search expressiveness.
   - `MaxAbsScaler` is relevant for sparse features.

8. `Binarizer`, `ZeroCount`
   - Cheap features for sparse/binary-heavy data.
   - Lower priority unless sparse/categorical pipeline expansion is planned.

### Lower priority / not recommended first

- `GaussianProcessRegressor`: expensive, poor scaling for common AutoML tabular sizes.
- Separate `xgb_limitdepth`: useful FLAML search trick, but better represented as XGBoost hyperparameter preset/search-space rule.
- Full H2O backend models: dependency/runtime mismatch with FEDOT. Prefer Python equivalents.
- AutoGluon `FASTAI`: useful but dependency-heavy and overlaps with `TabM`/`RealMLP`.
- AutoML-level `StackedEnsemble` as a model operation: conceptually conflicts with FEDOT's graph/composition layer. Better improve ensemble planning/composer behavior separately.

## Suggested implementation order for FEDOT

1. Low-risk sklearn additions:
   - `ExtraTreesClassifier`
   - `HistGradientBoostingClassifier/Regressor`
   - `GaussianNB`
   - `ElasticNet`
   - `MLPRegressor`

2. Low-risk preprocessing additions:
   - `VarianceThreshold`
   - `SelectPercentile` / `SelectFwe`
   - `TruncatedSVD`
   - explicit `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`

3. Categorical encoders:
   - `frequency_encoding`
   - leakage-safe `target_encoding`
   - rare-category grouping before OHE

4. Neural optional pack:
   - `TabM`
   - `RealMLP`
   - newer `TabPFNv2`/`TabDPT`/`Mitra` only if dependency and resource policy is agreed.

5. Advanced transformations:
   - `Nystroem`
   - `RBFSampler`
   - `FeatureAgglomeration`
   - `RandomTreesEmbedding`
   - `SelectFromModel`

## FEDOT integration notes

- Add model/operation metadata first in repository JSON, then implementation strategy mapping. Keep `Fedot` API as orchestration shell.
- For new deterministic inclusion/exclusion decisions, prefer typed rule functions/plans instead of inline API branching.
- Do not put leakage-prone target encoding into a simple transform without fold-aware fit semantics.
- Neural/foundation models should be `non-default` initially unless resource gates are explicit. Use tags such as `neural`, `non-default`, `cpu`/`gpu`, and possibly dataset-size rules.
- For repeated model variants that differ only by hyperparameter policy (`xgb_limitdepth`, L1/L2 logistic), prefer default-params/search-space changes over new operation names unless composition needs to distinguish them.
- Add direct deterministic tests for repository visibility and strategy mapping. For preprocessing, add transform shape/type tests and leakage/regression tests for target encoding.

## Source links

- AutoGluon Tabular fit/presets docs: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
- FLAML docs: https://microsoft.github.io/FLAML/
- auto-sklearn manual: https://automl.github.io/auto-sklearn/master/manual.html
- auto-sklearn classification components: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification
- auto-sklearn regression components: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression
- auto-sklearn feature preprocessors: https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/feature_preprocessing
- TPOT classifier config: https://raw.githubusercontent.com/EpistasisLab/tpot/master/tpot/config/classifier.py
- TPOT regressor config: https://raw.githubusercontent.com/EpistasisLab/tpot/master/tpot/config/regressor.py
- H2O AutoML docs: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
