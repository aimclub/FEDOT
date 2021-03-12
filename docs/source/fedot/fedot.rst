Intro to FEDOT
==============

FEDOT - an open-source framework for automated modeling and machine learning (AutoML). It can build custom modeling pipelines for different real-world processes in an automated way using an evolutionary approach. FEDOT supports classification (binary and multiclass), regression, clustering, and time series prediction tasks.

The framework is not limited to specific AutoML tasks (such as pre-processing of input data, feature selection, or optimization of model hyperparameters), but allows you to solve a more general structural learning problem - for a given data set, a solution is built in the form of a graph (DAG), the nodes of which are represented by ML models, pre-processing procedures, and data transformation.


FEDOT features
==============

The main features of the framework are as follows:

- The FEDOT architecture is highly flexible and therefore the framework can be used to automate the creation of mathematical models for various problems, types of data, and models;
- FEDOT already supports popular ML libraries (scikit-learn, keras, statsmodels, etc.), but you can also integrate custom tools into the framework if necessary;
- Pipeline optimization algorithms are not tied to specific data types or tasks, but you can use special templates for a specific task class or data type (time series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- The framework is not limited only to machine learning, it is possible to embed models related to specific areas into pipelines (for example, models in ODE or PDE);
- Additional methods for hyperparameters tuning can be seamlessly integrated into FEDOT (in addition to those already supported);
- The resulting pipelines can be exported in a human-readable JSON format, which allows you to achieve reproducibility of the experiments.

Thus, compared to other frameworks, FEDOT:

- Is not limited to specific modeling tasks and claims versatility and expandability;
- Supports the the complex modelling pipelines with variable shape and structure;
- Allows building models using input data of various nature (texts, images, tables, etc.) and consisting of different types of models.
- Allows managing the complexity of models and thereby achieving better results.