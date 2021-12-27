import os
import sys
from subprocess import Popen
import subprocess
import pandas as pd

# for correct .bat files execution fedot should be build as a package in environment
env_name = sys.executable
env_path_placeholder = 'DEFAULT'
predictions_path = '../../fedot/api/predictions.csv'


def change_python_path(file_name, old, new):
    """ Function for changing env path in .bat for users settings"""
    with open(file_name, "r+") as file:
        text = file.read()
        file.seek(0)
        text = text.replace(old, new)
        file.write(text)
        file.truncate()


def run_console(bat_name):
    """ Function for running .bat files with returning prediction as df"""
    try:
        os.remove(predictions_path)
    except Exception:
        pass
    change_python_path(bat_name, env_path_placeholder, env_name)
    process = Popen(bat_name, creationflags=subprocess.CREATE_NEW_CONSOLE)
    process.wait()
    change_python_path(bat_name, env_name, env_path_placeholder)
    print(f"\nPrediction saved at {predictions_path}")
    df = pd.read_csv(predictions_path)
    return df


def run_cli_ts_forecasting():
    """ Test executing ts_forecasting problem from cli with saving prediction"""
    bat_name = 'cli_ts_call.bat'
    run_console(bat_name)


def run_cli_classification():
    """ Test executing classification problem from cli with saving prediction"""
    bat_name = 'cli_classification_call.bat'
    run_console(bat_name)


def run_cli_regression():
    """ Test executing regression problem from cli with saving prediction"""
    bat_name = 'cli_regression_call.bat'
    run_console(bat_name)


if __name__ == '__main__':
    run_cli_classification()
