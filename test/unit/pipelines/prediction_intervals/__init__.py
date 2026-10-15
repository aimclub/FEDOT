import pathlib
import platform

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath
