set python_path = "DEFAULT"
cd ../../../cli_application
%python_path% fedot_cli.py --problem regression --train ../test/data/simple_regression_train.csv --test ../test/data/simple_regression_test.csv --timeout 0.1