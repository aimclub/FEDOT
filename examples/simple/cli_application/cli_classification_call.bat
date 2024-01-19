set python_path = "DEFAULT"
cd ../../fedot/api
%python_path% fedot_cli.py --problem classification --train ../../test/data/classification/simple_classification.csv --test ../../test/data/classification/simple_classification.csv  --target Y --timeout 0.1
