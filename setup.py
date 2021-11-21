from setuptools import find_packages, setup

with open('data_reconstruct/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='data_reconstruct',
    version=__version__,
    packages=find_packages(exclude=('tests')),
    description=readme,
    test_suite='tests',
    install_requires=[
        'matplotlib',
        'numpy',
        'sklearn',
        'torch',
        'unioncom',
    ],
    extras_require={
        'dev': [
            'flake8',
            'flake8-docstrings',
            'flake8-import-order',
            'jupyterlab',
            'pip-tools',
            'pytest',
            'pytest-cov',
            'pytest-mock',
        ],
    },
)
