import os
import sys
from subprocess import Popen
import subprocess

env_name = sys.executable
print(env_name)
with open('cli_ts_call.bat', "r+") as file:
    text = file.read()
    file.seek(0)
    text = text.replace('DEFAULT', env_name)
    file.write(text)
    file.truncate()

process = Popen('cli_ts_call.bat', creationflags=subprocess.CREATE_NEW_CONSOLE)
process.wait()
with open('cli_ts_call.bat') as file:
    text = file.read()
    file.seek(0)
    text = text.replace(env_name, 'DEFAULT')
    file.write(text)
    file.truncate()
