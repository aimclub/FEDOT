import os
import sys
from os.path import dirname, join
from pathlib import Path
from shutil import rmtree

from setuptools import setup, find_packages, Command

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
README = (Path(os.path.join(HERE, "README.rst"))).read_text()
NAME = "fedot"
AUTHOR = "NSS Lab"
VERSION = "0.1.0"
SHORT_DESCRIPTION = "Evolutionary structural learning framework FEDOT"
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
    extra_requirements = extract_requirements('extra_requirements.txt')
    requirements.extend(extra_requirements)
    return requirements


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(HERE, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(HERE, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(f'{sys.executable} setup.py sdist bdist_wheel --universal')

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system(f"git tag v{about['__version__']}")
        os.system('git push --tags')

        sys.exit()


setup_kwargs = dict(
    name=NAME,
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type="text/x-rst",
    url=URL,
    author=AUTHOR,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "fedot=fedot.__main__:main",
        ]
    },
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)

if __name__ == '__main__':
    setup(**setup_kwargs)
