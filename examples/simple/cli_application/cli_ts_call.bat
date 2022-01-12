set python_path = "DEFAULT"
cd ../../fedot/api
%python_path% fedot_cli.py --problem ts_forecasting --train ../../test/data/simple_time_series.csv --test ../../test/data/simple_time_series.csv --for_len 10 --target sea_height --c_timeout 0.1