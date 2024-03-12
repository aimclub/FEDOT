FEDOT Command Line Interface (CLI)
==================================

FEDOT API can be called via console without writing python code.
All API parameters can be marked as flags described in application Help.
Prediction saves as a CSV file for future use.


For correct console application run, there should be Python environment with installed
FEDOT package with all dependencies.


*Pay attention that your FEDOT working project can differ from the package version installed in the environment!*
*For setting master version as a package in environment download it through pip from GitHub with command:*

``pip install git+https://github.com/aimclub/FEDOT.git``

Start using
-----------

The main executing script running the application is fedot_cli.py so first, there is a need to navigate to them:


``cd {path_to_fedot}/fedot/api``

To get a list of possible flags and their descriptions, a help call is provided:

``python fedot_cli.py --help``

The result of execution is presented below:

.. code-block:: console

  --problem PROBLEM     The name of modelling problem to solve:
                        classification;
                        regression;
                        ts_forecasting;
                        clustering
  --train TRAIN         Path to train data file
  --test TEST           Path to test data file
  --preset PRESET       Name of preset for model building:
                        light;
                        light_steady_state;
                        ultra_light;
                        ultra_steady_state;
                        ts;
                        gpu
  --timeout TIMEOUT     Time for model design (in minutes)
  --seed SEED           Value for fixed random seed
  --target TARGET       Name of target variable in data
  --depth DEPTH         Composer parameter: max depth of the pipeline
  --arity ARITY         Composer parameter: max arity of the pipeline nodes
  --popsize POPSIZE     Composer parameter: population size
  --gen_num GEN_NUM     Composer parameter: number of generations
  --opers [OPERS [OPERS ...]]
                        Composer parameter: model names to use
  --tuning TUNING       Composer parameter: 1 - with tuning, 0 - without tuning
  --cv_folds CV_FOLDS   Composer parameter: Number of folds for cross-validation
  --hist_path HIST_PATH
                        Composer parameter: Name of the folder for composing history
  --for_len FOR_LEN     Time Series Forecasting parameter: forecast length

Examples of using (.bat files)
------------------------------

Examples of usage can be presented as .bat files for console execution. These files are located at
``/examples/cli_application`` folder. There the templates of parameters for different
problems decision are presented.

The string below helps to run classification problem decision from the console:

``python --problem classification --train ../../test/data/simple_classification.csv --test ../../test/data/simple_classification.csv  --target Y --timeout 0.1``
