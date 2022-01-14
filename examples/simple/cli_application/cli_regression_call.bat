set python_path = "C:\Users\yulas\Documents\fedot\venv\Scripts\python.exe"
cd ../../fedot/api
%python_path% fedot_cli.py --problem regression --train ../../test/data/simple_regression_train.csv --test ../../test/data/simple_regression_test.csv --c_timeout 0.1