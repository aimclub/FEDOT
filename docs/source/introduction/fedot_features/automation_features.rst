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

.. csv-table:: Feature transformation operations definitions
   :header: "API name","Definition", "Problem"

   `rfe_lin_reg`,Linear Regression Recursive Feature Elimination, Feature extraction
   `rfe_non_lin_reg`,Decision Tree Recursive Feature Elimination, Feature extraction
   `rfe_lin_class`,Logistic Regression Recursive Feature Elimination, Feature extraction
   `rfe_non_lin_class`,Decision Tree Recursive Feature Elimination, Feature extraction
   `isolation_forest_reg`,Regression Isolation Forest, Regression anomaly detection
   `isolation_forest_class`,Classification Isolation Forest, Classification anomaly detection
   `ransac_lin_reg`,Regression Random Sample Consensus, Outlier detection
   `ransac_non_lin_reg`,Decision Tree Random Sample Consensus, Outlier detection
   `pca`,Principal Component Analysis, Dimensionality reduction
   `kernel_pca`,Kernel Principal Component Analysis, Dimensionality reduction
   `fast_ica`,Independent Component Analysis, Feature extraction
   `poly_features`,Polynomial Features, Feature engineering
   `decompose`,Diff regression prediction and target for new target, Feature extraction
   `class_decompose`,Diff classification prediction and target for new target, Feature extraction
   `cntvect`,Count Vectorizer, Text feature extraction
   `text_clean`,Lemmatization and Stemming, Text data processing
   `tfidf`,TF-IDF Vectorizer, Text feature extraction
   `word2vec_pretrained`,Text vectorization, Text feature extraction
   `lagged`,Time series to the Hankel matrix transformation, Time series transformation
   `sparse_lagged`,As `lagged` but with sparsing, Time series transformation
   `smoothing`,Moving average, Time series transformation
   `gaussian_filter`,Gaussian Filter, Time series transformation
   `diff_filter`,Derivative Filter, Time series transformation
   `cut`,Cut timeseries, Timeseries transformation
   `scaling`,Scaling, Feature scaling
   `normalization`,Normalization, Feature normalization
   `simple_imputation`,Imputation, Data imputation
   `one_hot_encoding`,One-Hot Encoder, Feature encoding
   `label_encoding`,Label Encoder, Feature encoding
   `resample`,Imbalanced binary class transformation in classification, Data transformation
   `topological_features`,Calculation of topological features,Time series transformation
   `dummy`,Simple forwarding operator in pipeline, Data forwarding


.. csv-table:: Feature transformation operations implementations
   :header: "API name","Model used","Presets"

   `rfe_lin_reg`,`sklearn.feature_selection.RFE`, 
   `rfe_non_lin_reg`,`sklearn.feature_selection.RFE`,
   `rfe_lin_class`,`sklearn.feature_selection.RFE`,
   `rfe_non_lin_class`,`sklearn.feature_selection.RFE`,
   `isolation_forest_reg`,`sklearn.ensemble.IsolationForest`,
   `isolation_forest_class`,`sklearn.ensemble.IsolationForest`,
   `ransac_lin_reg`,`sklearn.linear_model.RANSACRegressor`,`fast_train` `*tree`
   `ransac_non_lin_reg`,`sklearn.linear_model.RANSACRegressor`, `*tree`
   `pca`,`sklearn.decomposition.PCA`,`fast_train` `ts` `*tree`
   `kernel_pca`,`sklearn.decomposition.KernelPCA`,`ts` `*tree`
   `fast_ica`,`sklearn.decomposition.FastICA`,`ts` `*tree`
   `poly_features`,`sklearn.preprocessing.PolynomialFeatures`,
   `decompose`,`FEDOT model`,`fast_train` `ts` `*tree`
   `class_decompose`,`FEDOT model`,`fast_train` `*tree`
   `cntvect`,`sklearn.feature_extraction.text.CountVectorizer`,
   `text_clean`,`nltk.stem.WordNetLemmatizer nltk.stem.SnowballStemmer`,
   `tfidf`,`sklearn.feature_extraction.text.TfidfVectorizer`,
   `word2vec_pretrained`,`Gensin-data model <https://github.com/piskvorky/gensim-data>`_,
   `lagged`,`FEDOT model`,`fast_train` `ts`
   `sparse_lagged`,`FEDOT model`,`fast_train` `ts`
   `smoothing`,`FEDOT model`,`fast_train` `ts`
   `gaussian_filter`,`FEDOT model`,`fast_train` `ts`
   `diff_filter`,`FEDOT model`,`ts`
   `cut`,`FEDOT model`,`fast_train` `ts`
   `scaling`,`sklearn.preprocessing.StandardScaler`,`fast_train` `ts` `*tree`
   `normalization`,`sklearn.preprocessing.MinMaxScaler`,`fast_train` `ts` `*tree`
   `simple_imputation`,`sklearn.impute.SimpleImputer`,`fast_train` `*tree`
   `one_hot_encoding`,`sklearn.preprocessing.OneHotEncoder`,
   `label_encoding`,`sklearn.preprocessing.LabelEncoder`,`fast_train` `*tree`
   `resample`,`FEDOT model using sklearn.utils.resample`,
   `topological_features`,FEDOT model,`ts`,
   `fast_topological_features`,FEDOT model,`ts`
   `dummy`,FEDOT model,`fast_train` `*tree`


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


.. csv-table:: Available models definitions
   :header: "API name","Definition","Problem"

   `adareg`,AdaBoost regressor,Regression
   `blendreg`, Weighted Average Blending,Regression
   `catboostreg`,Catboost regressor,Regression
   `cbreg_bag`, Catboost Bagging regressor,Regression
   `dtreg`,Decision Tree regressor,Regression
   `gbr`,Gradient Boosting regressor,Regression
   `knnreg`,K-nearest neighbors regressor,Regression
   `lasso`,Lasso Linear regressor,Regression
   `lgbmreg`,Light Gradient Boosting Machine regressor,Regression
   `lgbmreg_bag`,Light Gradient Boosting Machine Bagging regressor,Regression
   `linear`,Linear Regression regressor,Regression
   `rfr`,Random Forest regressor,Regression
   `ridge`,Ridge Linear regressor,Regression
   `sgdr`,Stochastic Gradient Descent regressor,Regression
   `svr`,Linear Support Vector regressor,Regression
   `treg`,Extra Trees regressor,Regression
   `xgboostreg`,Extreme Gradient Boosting regressor,Regression
   `xgbreg_bag`,Extreme Gradient Boosting Bagging regressor,Regression
   `bernb`,Naive Bayes classifier (multivariate Bernoulli),Classification
   `blending`, Weighted Average Blending,Classification
   `catboost`,Catboost classifier,Classification
   `cb_bag`, Catboost Bagging classifier,Classification
   `cnn`,Convolutional Neural Network,Classification
   `dt`,Decision Tree classifier,Classification
   `knn`,K-nearest neighbors classifier,Classification
   `lda`,Linear Discriminant Analysis,Classification
   `lgbm`,Light Gradient Boosting Machine classifier,Classification
   `lgbm_bag`,Light Gradient Boosting Machine Bagging classifier,Classification
   `logit`,Logistic Regression classifier,Classification
   `mlp`,Multi-layer Perceptron classifier,Classification
   `multinb`,Naive Bayes classifier (multinomial),Classification
   `qda`,Quadratic Discriminant Analysis,Classification
   `rf`,Random Forest classifier,Classification
   `svc`,Support Vector classifier,Classification
   `xgboost`,Extreme Gradient Boosting classifier,Classification
   `xgb_bag`,Extreme Gradient Boosting Bagging classifier,Classification
   `kmeans`,K-Means clustering,Clustering
   `ar`,AutoRegression,Forecasting
   `arima`,ARIMA,Forecasting
   `cgru`,Convolutional Gated Recurrent Unit,Forecasting
   `clstm`,Convolutional Long Short-Term Memory,Forecasting
   `ets`,Exponential Smoothing,Forecasting
   `glm`,Generalized Linear Models,Forecasting
   `locf`,Last Observation Carried Forward,Forecasting
   `polyfit`,Polynomial approximation,Forecasting
   `stl_arima`,STL Decomposition with ARIMA,Forecasting
   `ts_naive_average`,Naive Average,Forecasting


.. csv-table:: Available models implementations
   :header: "API name","Model used","Presets"

   `adareg`,`sklearn.ensemble.AdaBoostRegressor`,`fast_train` `ts` `*tree`
   `blendreg`, `FEDOT model`, `*tree`
   `catboostreg`,`catboost.CatBoostRegressor`,`*tree`
   `cbreg_bag`,`sklearn.ensemble.BaggingRegressor and catboost.CatBoostRegressor`,`*tree`
   `dtreg`,`sklearn.tree.DecisionTreeRegressor`,`fast_train` `ts` `*tree`
   `gbr`,`sklearn.ensemble.GradientBoostingRegressor`,`*tree`
   `knnreg`,`sklearn.neighbors.KNeighborsRegressor`,`fast_train` `ts`
   `lasso`,`sklearn.linear_model.Lasso`,`fast_train` `ts`
   `lgbmreg`,`lightgbm.sklearn.LGBMRegressor`,`*tree`
   `lgbmreg_bag`,``sklearn.ensemble.BaggingRegressor and lightgbm.sklearn.LGBMRegressor`,`*tree`
   `linear`,`sklearn.linear_model.LinearRegression`,`fast_train` `ts`
   `rfr`,`sklearn.ensemble.RandomForestRegressor`,`fast_train` `*tree`
   `ridge`,`sklearn.linear_model.Ridge`,`fast_train` `ts`
   `sgdr`,`sklearn.linear_model.SGDRegressor`,`fast_train` `ts`
   `svr`,`sklearn.svm.LinearSVR`,
   `treg`,`sklearn.ensemble.ExtraTreesRegressor`,`*tree`
   `xgboostreg`,`xgboost.XGBRegressor`,`*tree`
   `xgbreg_bag`,`sklearn.ensemble.BaggingRegressor and xgboost.XGBRegressor`,`*tree`
   `bernb`,`sklearn.naive_bayes.BernoulliNB`,`fast_train`
   `blending`, `FEDOT model`, `*tree`
   `catboost`,`catboost.CatBoostClassifier`,`*tree`
   `cb_bag`,`sklearn.ensemble.BaggingClassifier and catboost.CatBoostClassifier`,`*tree`
   `cnn`,`FEDOT model`,
   `dt`,`sklearn.tree.DecisionTreeClassifier`,`fast_train` `*tree`
   `knn`,`sklearn.neighbors.KNeighborsClassifier`,`fast_train`
   `lda`,`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`,`fast_train`
   `lgbm`,`lightgbm.sklearn.LGBMClassifier`,
   `lgbm_bag`,`sklearn.ensemble.BaggingClassifier and lightgbm.sklearn.LGBMClassifier`, `*tree`
   `logit`,`sklearn.linear_model.LogisticRegression`,`fast_train`
   `mlp`,`sklearn.neural_network.MLPClassifier`,
   `multinb`,`sklearn.naive_bayes.MultinomialNB`,`fast_train`
   `qda`,`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`,`fast_train`
   `rf`,`sklearn.ensemble.RandomForestClassifier`,`fast_train` `*tree`
   `svc`,`sklearn.svm.SVC`,
   `xgboost`,`xgboost.XGBClassifier`,`*tree`
   `xgb_bag`,`sklearn.ensemble.BaggingClassifier and xgboost.XGBClassifier`,`*tree`
   `kmeans`,`sklearn.cluster.Kmeans`,`fast_train`
   `ar`,`statsmodels.tsa.ar_model.AutoReg`,`fast_train` `ts`
   `arima`,`statsmodels.tsa.arima.model.ARIMA`,`ts`
   `cgru`,`FEDOT model`,`ts`
   `clstm`,`FEDOT model`,`ts`
   `ets`,`statsmodels.tsa.exponential_smoothing.ets.ETSModel`,`fast_train` `ts`
   `glm`,`statsmodels.genmod.generalized_linear_model.GLM`,`fast_train` `ts`
   `locf`,`FEDOT model`,`fast_train` `ts`
   `polyfit`,`FEDOT model`,`fast_train` `ts`
   `stl_arima`,`statsmodels.tsa.api.STLForecast`,`ts`
   `ts_naive_average`,`FEDOT model`,`fast_train` `ts`
