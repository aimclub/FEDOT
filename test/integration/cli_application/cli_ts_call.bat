set python_path = "C:\Users\yulas\.conda\envs\fedot_env\python.exe"
cd ../../../cli_application
%python_path% fedot_cli.py --problem ts_forecasting --train ../test/data/simple_time_series.csv --test ../test/data/simple_time_series.csv --for_len 10 --target sea_height --timeout 0.3