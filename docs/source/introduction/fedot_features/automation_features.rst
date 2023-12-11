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

These algorithms are used for filtering and selecting specific features.

.. csv-table:: Feature selection operations
   :header: "API name","Model used","Definition"

   `rfe_lin_reg`,`sklearn.feature_selection.RFE`,Linear Regression Recursive Feature Elimination
   `rfe_non_lin_reg`,`sklearn.feature_selection.RFE`,Decision Tree Recursive Feature Elimination
   `rfe_lin_class`,`sklearn.feature_selection.RFE`,Logistic Regression Recursive Feature Elimination
   `rfe_non_lin_class`,`sklearn.feature_selection.RFE`,Decision Tree Recursive Feature Elimination
   `isolation_forest_reg`,`sklearn.ensemble.IsolationForest`,Regression Isolation Forest
   `isolation_forest_class`,`sklearn.ensemble.IsolationForest`,Classification Isolation Forest
   `ransac_lin_reg`,`sklearn.linear_model.RANSACRegressor`,Regression Random Sample Consensus
   `ransac_non_lin_reg`,`sklearn.linear_model.RANSACRegressor`,DecisionTreeRegressor Random Sample Consensus


Feature extraction
""""""""""""""""""

These algorithms are used for generating new features.

.. csv-table:: Feature extraction operations
   :header: "API name","Model used","Definition"

   `pca`,`sklearn.decomposition.PCA`,Principal Component Analysis
   `kernel_pca`,`sklearn.decomposition.KernelPCA`,Kernel Principal Component Analysis
   `fast_ica`,`sklearn.decomposition.FastICA`,Independent Component Analysis
   `poly_features`,`sklearn.preprocessing.PolynomialFeatures`,Polynomial Features
   `decompose`,`DecomposerRegImplementation`,Regression Decomposition
   `class_decompose`,`DecomposerClassImplementation`,Classification Decomposition
   `cntvect`,`sklearn.feature_extraction.text.CountVectorizer`,Count Vectorizer
   `text_clean`,`nltk.stem.WordNetLemmatizer nltk.stem.SnowballStemmer`,Lemmatization and Stemming
   `tfidf`,`sklearn.feature_extraction.text.TfidfVectorizer`,TF-IDF Vectorizer
   `word2vec_pretrained`,`PretrainedEmbeddingsImplementation`,Word2Vec
   `lagged`,`LaggedTransformationImplementation`,Lagged timeseries tranformation
   `sparse_lagged`,`SparseLaggedTransformationImplementation`,Sparse Lagged timeseries tranformation
   `smoothing`,`TsSmoothingImplementation`,Smoothing timeseries tranformation
   `gaussian_filter`,`GaussianFilterImplementation`,Gaussian Filter timeseries tranformation
   `diff_filter`,`NumericalDerivativeFilterImplementation`,Derivative Filter timeseries tranformation
   `cut`,`CutImplementation`,Cut timeseries tranformation
   `exog_ts`,`ExogDataTransformationImplementation`,Exogeneus timeseries tranformation


Feature expansion
"""""""""""""""""

These methods are used for expanding specific features to a bigger amount or changing their bounds.

.. csv-table:: Feature expansion operations
   :header: "API name","Model used","Definition"

   `scaling`,`sklearn.preprocessing.StandardScaler`,Scaling
   `normalization`,`sklearn.preprocessing.MinMaxScaler`,Normalization
   `simple_imputation`,`sklearn.impute.SimpleImputer`,Imputation
   `one_hot_encoding`,`sklearn.preprocessing.OneHotEncoder`,Ohe-Hot Encoder
   `label_encoding`,`sklearn.preprocessing.LabelEncoder`,Label Encoder
   `resample`,`ResampleImplementation`,Resample features


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
   :header: "API name","Model used","Definition","Problem"

   `adareg`,`sklearn.ensemble.AdaBoostRegressor`,AdaBoost regressor,Regression
   `catboostreg`,`catboost.CatBoostRegressor`,Catboost regressor,Regression
   `dtreg`,`sklearn.tree.DecisionTreeRegressor`,Decision Tree regressor,Regression
   `gbr`,`sklearn.ensemble.GradientBoostingRegressor`,Gradient Boosting regressor,Regression
   `knnreg`,`sklearn.neighbors.KNeighborsRegressor`,K-nearest neighbors regressor,Regression
   `lasso`,`sklearn.linear_model.Lasso`,Lasso Linear regressor,Regression
   `lgbmreg`,`lightgbm.sklearn.LGBMRegressor`,Light Gradient Boosting Machine regressor,Regression
   `linear`,`sklearn.linear_model.LinearRegression`,Linear Regression regressor,Regression
   `rfr`,`sklearn.ensemble.RandomForestRegressor`,Random Forest regressor,Regression
   `ridge`,`sklearn.linear_model.Ridge`,Ridge Linear regressor,Regression
   `sgdr`,`sklearn.linear_model.SGDRegressor`,Stochastic Gradient Descent regressor,Regression
   `svr`,`sklearn.svm.LinearSVR`,Linear Support Vector regressor,Regression
   `treg`,`sklearn.ensemble.ExtraTreesRegressor`,Extra Trees regressor,Regression
   `xgbreg`,`xgboost.XGBRegressor`,Extreme Gradient Boosting regressor,Regression
   `bernb`,`sklearn.naive_bayes.BernoulliNB`,Naive Bayes classifier (multivariate Bernoulli),Classification
   `catboost`,`catboost.CatBoostClassifier`,Catboost classifier,Classification
   `cnn`,`FedotCNNImplementation`,Convolutional Neural Network,Classification
   `dt`,`sklearn.tree.DecisionTreeClassifier`,Decision Tree classifier,Classification
   `knn`,`sklearn.neighbors.KNeighborsClassifier`,K-nearest neighbors classifier,Classification
   `lda`,`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`,Linear Discriminant Analysis,Classification
   `lgbm`,`lightgbm.sklearn.LGBMClassifier`,Light Gradient Boosting Machine classifier,Classification
   `logit`,`sklearn.linear_model.LogisticRegression`,Logistic Regression classifier,Classification
   `mlp`,`sklearn.neural_network.MLPClassifier`,Multi-layer Perceptron classifier,Classification
   `multinb`,`sklearn.naive_bayes.MultinomialNB`,Naive Bayes classifier (multinomial),Classification
   `qda`,`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`,Quadratic Discriminant Analysis,Classification
   `rf`,`sklearn.ensemble.RandomForestClassifier`,Random Forest classifier,Classification
   `svc`,`sklearn.svm.SVC`,Support Vector classifier,Classification
   `xgboost`,`xgboost.XGBClassifier`,Extreme Gradient Boosting classifier,Classification
   `kmeans`,`sklearn.cluster.Kmeans`,K-Means clustering,Clustering
   `ar`,`statsmodels.tsa.ar_model.AutoReg`,AutoRegression,Forecasting
   `arima`,`statsmodels.tsa.arima.model.ARIMA`,ARIMA,Forecasting
   `cgru`,`CGRUImplementation`,Convolutional Gated Recurrent Unit,Forecasting
   `ets`,`statsmodels.tsa.exponential_smoothing.ets.ETSModel`,Exponential Smoothing,Forecasting
   `glm`,`statsmodels.genmod.generalized_linear_model.GLM`,Generalized Linear Models,Forecasting
   `locf`,`RepeatLastValueImplementation`,Last Observation Carried Forward,Forecasting
   `polyfit`,`PolyfitImplementation`,Polynomial fitter,Forecasting
   `stl_arima`,`statsmodels.tsa.api.STLForecast`,STL Decomposition with ARIMA,Forecasting
   `ts_naive_average`,`NaiveAverageForecastImplementation`,Naive Average,Forecasting
