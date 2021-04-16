import os
from os.path import dirname, join
from pathlib import Path

import setuptools

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
NAME = "fedot"
VERSION = "0.3.0"
AUTHOR = "NSS Lab"
SHORT_DESCRIPTION = "Evolutionary structural learning framework FEDOT"
README = (Path(os.path.join(HERE, "README.rst"))).read_text()
URL = "https://github.com/nccr-itmo/FEDOT"
REQUIRES_PYTHON = '>=3.6'
LICENSE = "BSD 3-Clause"


def read(*names, **kwargs):
    with open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


def extract_requirements(file_name):
    return [r for r in read(file_name).split('\n') if r and not r.startswith('#')]


def get_requirements():
    requirements = extract_requirements('requirements.txt')
    return requirements


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email="itmo.nss.team@gmail.com",
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type="text/x-rst",
    url=URL,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=setuptools.find_packages(exclude=['test*']),
    include_package_data=True,
    install_requires=get_requirements(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
