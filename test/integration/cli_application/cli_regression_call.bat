set python_path = "DEFAULT"
cd ../../../fedot/api
%python_path% fedot_cli.py --problem regression --train ../../test/data/simple_regression_train.csv --test ../../test/data/simple_regression_test.csv --c_timeout 0.1