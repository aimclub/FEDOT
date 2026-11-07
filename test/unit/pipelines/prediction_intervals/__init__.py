import pathlib
import platform

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
elif plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
