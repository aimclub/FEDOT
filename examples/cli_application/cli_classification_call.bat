set python_path = "DEFAULT"
cd ../../fedot/api
%python_path% fedot_cli.py --problem classification --train ../../test/data/simple_classification.csv --test ../../test/data/simple_classification.csv  --target Y --c_timeout 0.1