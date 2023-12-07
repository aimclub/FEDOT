AutoML capabilities
-------------------

FEDOT is capable of setting its 'automation rate' by omitting some of its parameters.
For example, if you just create the FEDOT instance and call the ``fit`` method with the appropriate dataset on it,
you will have a full automation of the learning process,
see :doc:`automated composing </introduction/tutorial/composing_pipelines/automated_creation>`

At the same time, if you pass some of the parameters, you will have a partial automation,
see :doc:`manual composing </introduction/tutorial/composing_pipelines/manual_creation>`.

Other than that, you can utilize more AutoML means.


Data nature
^^^^^^^^^^^

FEDOT uses specific data processing according to the source
of the input data (whether it is pandas DataFrame, or numpy array, or just a path to the dataset, etc).

.. note::

    Be careful with datetime features, as they are to be casted into a float type with milliseconds unit.


Apart from that FEDOT is capable of working with multi-modal data.
It means that you can pass it different types of datasets
(tables, texts, images, etc) and it will get the information from within to work with it.

.. seealso::
    :doc:`Detailed multi-modal data description and usage </basics/multi_modal_tasks>`


Dimensional operations
^^^^^^^^^^^^^^^^^^^^^^

FEDOT supports bunch of dimensionality preprocessing operations that can be be added to the pipeline as a node.

Feature selection
"""""""""""""""""

There are different linear and non-linear algorithms for regression and classification tasks
which uses scikit-learn's Recursive Feature Elimination (RFE).

.. list-table:: Feature selection operations
   :header-rows: 1

   * - API name
     - Definition
   * - rfe_lin_reg
     - RFE via Linear Regression Regressor
   * - rfe_non_lin_reg
     - RFE via Decision tree Regressor
   * - rfe_lin_class
     - RFE via Logistic Regression Classifier
   * - rfe_non_lin_class
     - RFE via Decision tree Classifier

Feature extraction
""""""""""""""""""

These algorithms are used for generating new features.

.. list-table:: Feature extraction operations
   :header-rows: 1

   * - API name
     - Definition
   * - pca
     - Principal Component Analysis (PCA)
   * - kernel_pca
     - Principal Component Analysis (PCA) with kernel methods
   * - fast_ica
     - Fast Independent Component Analysis (FastICA)
   * - poly_features
     - Polynomial Features transformations
   * - lagged
     - Time-series to table transformation
   * - sparse_lagged
     - Time-series to sparse table transformation

Feature expansion
"""""""""""""""""

These methods expands specific features to a bigger amount

.. list-table:: Feature expansion operations
   :header-rows: 1

   * - API name
     - Definition
   * - one_hot_encoding
     - One-hot encoding


Models used
^^^^^^^^^^^

Using the parameter ``preset`` of the :doc:`main API </api/api>` you can specify
what models are available during the learning process. 

It influences:

* composing speed and quality
* computational behaviour
* task relevance

For example, ``'best_quality'`` option allows FEDOT to use entire list of available models for a specified task.
In contrast ``'fast_train'`` ensures only fast learning models are going to be used.

Apart from that there are other options whose names speak for themselves: ``'stable'``, ``'auto'``, ``'gpu'``, ``'ts'``,
``'automl'`` (the latter uses only AutoML models as pipeline nodes).

.. note::
    To make it simple, FEDOT uses ``auto`` by default to identify the best choice for you.


.. csv-table:: Available models
   :header: "API name","Model used","Definition","Problem","Tags"

   `adareg`,`sklearn.ensemble.AdaBoostRegressor`,AdaBoost regressor,Regression,"`fast_train`, `ts`, `*tree`"
   `ar`,`statsmodels.tsa.ar_model.AutoReg`,AutoRegression,Forecasting,"`fast_train`, `ts`"
   `arima`,`statsmodels.tsa.arima.model.ARIMA`,ARIMA,Forecasting,"`ts`"
   `cgru`,`CGRUImplementation`,Convolutional Gated Recurrent Unit,Forecasting,"`ts`"
   `bernb`,`sklearn.naive_bayes.BernoulliNB`,Naive Bayes classifier (multivariate Bernoulli),Classification,"`fast_train`"
   `catboost`,`catboost.CatBoostClassifier`,Catboost classifier,Classification,"`*tree`"
   `catboostreg`,`catboost.CatBoostRegressor`,Catboost regressor,Regression,"`*tree`"
   `dt`,`sklearn.tree.DecisionTreeClassifier`,Decision Tree classifier,Classification,"`fast_train`, `*tree`"
   `dtreg`,`sklearn.tree.DecisionTreeRegressor`,Decision Tree regressor,Regression,"`fast_train`, `ts`, `*tree`"
   `gbr`,`sklearn.ensemble.GradientBoostingRegressor`,Gradient Boosting regressor,Regression,"`*tree`"
   `kmeans`,`sklearn.cluster.Kmeans`,K-Means clustering,Clustering,"`fast_train`"
   `knn`,`sklearn.neighbors.KNeighborsClassifier`,K-nearest neighbors classifier,Classification,"`fast_train`"
   `knnreg`,`sklearn.neighbors.KNeighborsRegressor`,K-nearest neighbors regressor,Regression,"`fast_train`, `ts`"
   `lasso`,`sklearn.linear_model.Lasso`,Lasso Linear regressor,Regression,"`fast_train`, `ts`"
   `lda`,`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`,Linear Discriminant Analysis,Classification,"`fast_train`"
   `lgbm`,`lightgbm.sklearn.LGBMClassifier`,Light Gradient Boosting Machine classifier,Classification,""
   `lgbmreg`,`lightgbm.sklearn.LGBMRegressor`,Light Gradient Boosting Machine regressor,Regression,"`*tree`"
   `linear`,`sklearn.linear_model.LinearRegression`,Linear Regression regressor,Regression,"`fast_train`, `ts`"
   `logit`,`sklearn.linear_model.LogisticRegression`,Logistic Regression classifier,Classification,"`fast_train`"
   `mlp`,`sklearn.neural_network.MLPClassifier`,Multi-layer Perceptron classifier,Classification,""
   `multinb`,`sklearn.naive_bayes.MultinomialNB`,Naive Bayes classifier (multinomial),Classification,"`fast_train`"
   `qda`,`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`,Quadratic Discriminant Analysis,Classification,"`fast_train`"
   `rf`,`sklearn.ensemble.RandomForestClassifier`,Random Forest classifier,Classification,"`fast_train`, `*tree`"
   `rfr`,`sklearn.ensemble.RandomForestRegressor`,Random Forest regressor,Regression,"`fast_train`, `*tree`"
   `ridge`,`sklearn.linear_model.Ridge`,Ridge Linear regressor,Regression,"`fast_train`, `ts`"
   `polyfit`,`PolyfitImplementation`,Polynomial fitter,Forecasting,"`fast_train`, `ts`"
   `sgdr`,`sklearn.linear_model.SGDRegressor`,Stochastic Gradient Descent regressor,Regression,"`fast_train`, `ts`"
   `stl_arima`,`statsmodels.tsa.api.STLForecast`,STL Decomposition with ARIMA,Forecasting,"`ts`"
   `glm`,`statsmodels.genmod.generalized_linear_model.GLM`,Generalized Linear Models,Forecasting,"`fast_train`, `ts`"
   `ets`,`statsmodels.tsa.exponential_smoothing.ets.ETSModel`,Exponential Smoothing,Forecasting,"`fast_train`, `ts`"
   `locf`,`RepeatLastValueImplementation`,Last Observation Carried Forward,Forecasting,"`fast_train`, `ts`"
   `ts_naive_average`,`NaiveAverageForecastImplementation`,Naive Average,Forecasting,"`fast_train`, `ts`"
   `svc`,`sklearn.svm.SVC`,Support Vector classifier,Classification,""
   `svr`,`sklearn.svm.LinearSVR`,Linear Support Vector regressor,Regression,""
   `treg`,`sklearn.ensemble.ExtraTreesRegressor`,Extra Trees regressor,Regression,"`*tree`"
   `xgboost`,`xgboost.XGBClassifier`,Extreme Gradient Boosting classifier,Classification,"`*tree`"
   `xgbreg`,`xgboost.XGBRegressor`,Extreme Gradient Boosting regressor,Regression,"`*tree`"
   `cnn`,`FedotCNNImplementation`,Convolutional Neural Network,Classification,""


Preprocessing operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table:: Preprocessing operations
   :header: "API name","Model used","Definition","Problem","Tags"

   `scaling`,`sklearn.preprocessing.StandardScaler`,Scaling,Feature Scaling,"`fast_train`, `ts`, `*tree`"
   `normalization`,`sklearn.preprocessing.MinMaxScaler`,Normalization,Feature Scaling,"`fast_train`, `ts`, `*tree`"
   `simple_imputation`,`sklearn.impute.SimpleImputer`,Imputation,Feature Imputation,"`fast_train`, `*tree`"
   `pca`,`sklearn.decomposition.PCA`,Principal Component Analysis,Feature Reduction,"`fast_train`, `ts`, `*tree`"
   `kernel_pca`,`sklearn.decomposition.KernelPCA`,Kernel Principal Component Analysis,Feature Reduction,"`ts`, `*tree`"
   `fast_ica`,`sklearn.decomposition.FastICA`,Independent Component Analysis,Feature Reduction,"`ts`, `*tree`"
   `poly_features`,`sklearn.preprocessing.PolynomialFeatures`,Polynomial Features,Feature Engineering,""
   `one_hot_encoding`,`sklearn.preprocessing.OneHotEncoder`,Ohe-Hot Encoder,Feature Encoding,""
   `label_encoding`,`sklearn.preprocessing.LabelEncoder`,Label Encoder,Feature Encoding,"`fast_train`, `*tree`"
   `rfe_lin_reg`,`sklearn.feature_selection.RFE`,Linear Regression Recursive Feature Elimination,Feature Selection,""
   `rfe_non_lin_reg`,`sklearn.feature_selection.RFE`,Decision Tree Recursive Feature Elimination,Feature Selection,""
   `rfe_lin_class`,`sklearn.feature_selection.RFE`,Logistic Regression Recursive Feature Elimination,Feature Selection,""
   `rfe_non_lin_class`,`sklearn.feature_selection.RFE`,Decision Tree Recursive Feature Elimination,Feature Selection,""
   `isolation_forest_reg`,`sklearn.ensemble.IsolationForest`,Regression Isolation Forest,Feature Filtering,""
   `isolation_forest_class`,`sklearn.ensemble.IsolationForest`,Classification Isolation Forest,Feature Filtering,""
   `decompose`,`DecomposerRegImplementation`,Regression Decomposition,Decomposition,"`fast_train`, `ts`, `*tree`"
   `class_decompose`,`DecomposerClassImplementation`,Classification Decomposition,Decomposition,"`fast_train`, `*tree`"
   `resample`,`ResampleImplementation`,Resample features,Resampling,""
   `ransac_lin_reg`,`sklearn.linear_model.RANSACRegressor`,Regression Random Sample Consensus,Feature Filtering,"`fast_train`, `*tree`"
   `ransac_non_lin_reg`,`sklearn.linear_model.RANSACRegressor`,DecisionTreeRegressor Random Sample Consensus,Feature Filtering,"`fast_train`, `*tree`"
   `cntvect`,`sklearn.feature_extraction.text.CountVectorizer`,Count Vectorizer,Text Processing,""
   `text_clean`,`nltk.stem.WordNetLemmatizer nltk.stem.SnowballStemmer`,Lemmatization and Stemming,Text Processing,""
   `tfidf`,`sklearn.feature_extraction.text.TfidfVectorizer`,TF-IDF Vectorizer,Text Processing,""
   `word2vec_pretrained`,`PretrainedEmbeddingsImplementation`,Word2Vec,Text Processing,""
   `lagged`,`LaggedTransformationImplementation`,Lagged Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `sparse_lagged`,`SparseLaggedTransformationImplementation`,Sparse Lagged Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `smoothing`,`TsSmoothingImplementation`,Smoothing Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `gaussian_filter`,`GaussianFilterImplementation`,Gaussian Filter Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `diff_filter`,`NumericalDerivativeFilterImplementation`,Derivative Filter Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `cut`,`CutImplementation`,Cut Tranformation,Timeseries Tranformation,"`fast_train`, `ts`"
   `exog_ts`,`ExogDataTransformationImplementation`,Exogeneus Tranformation,Timeseries Tranformation,""
