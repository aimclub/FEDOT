
Fedot CLI Execution Example
==========================

This example demonstrates how to execute Fedot tasks (time series forecasting, classification, and regression) from the command line interface (CLI) using .bat files. The code provided manipulates the Python environment path in the .bat files to ensure correct execution and saves the predictions to a CSV file.

Overview
--------

The example consists of several functions that handle the execution of .bat files for different Fedot tasks. The main functions are:

- `change_python_path`: Modifies the Python environment path in a .bat file.
- `run_console`: Executes a .bat file, waits for it to complete, and returns the predictions as a pandas DataFrame.
- `run_cli_ts_forecasting`, `run_cli_classification`, `run_cli_regression`: Specific functions to run .bat files for time series forecasting, classification, and regression tasks, respectively.

Step-by-Step Guide
------------------

1. **Import Necessary Libraries**
   ::

    import os
    import sys
    from subprocess import Popen
    import subprocess
    import pandas as pd

   This block imports the necessary Python libraries for file handling, subprocess execution, and data manipulation.

2. **Define Constants and Functions**
   ::

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

   - `env_name` stores the path to the Python executable.
   - `env_path_placeholder` is a placeholder used in the .bat files.
   - `predictions_path` is the path where predictions are saved.
   - `change_python_path` function replaces the Python environment path in a .bat file.
   - `run_console` function runs a .bat file, waits for it to finish, and returns the predictions as a DataFrame.

3. **Run Specific CLI Tasks**
   ::

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

   These functions specify the .bat files to be run for each task and call `run_console` to execute them.

4. **Main Execution Block**
   ::

    if __name__ == '__main__':
        run_cli_classification()

   This block ensures that the `run_cli_classification` function is called when the script is executed directly.

Usage
-----

To use this example, ensure you have the appropriate .bat files (`cli_ts_call.bat`, `cli_classification_call.bat`, `cli_regression_call.bat`) in the correct directory. Modify the `env_name` and `predictions_path` variables if necessary to match your environment and desired output path.

Run the script, and it will execute the specified .bat file, save the predictions to a CSV file, and print the location of the saved predictions.

.. note::
   Ensure that Fedot is built as a package in your environment for the .bat files to execute correctly.

.. seealso::
   For more information on Fedot and its CLI usage, refer to the `Fedot documentation <https://github.com/nccr-itmo/FEDOT>`_.