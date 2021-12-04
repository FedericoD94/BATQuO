# xg-lesioni

A library for QAOA with bayesian optimisation.

## Usage

### Poetry

Open the terminal and:
- Enter the repo: ```cd <path-to-this-repo-folder>```
- Install a poetry virtual environment: ```poetry install```
- Have an overview of the available arguments: ```poetry run python src/main_qutip.py --help```

NOTE: In order to install the poetry environment you **need Python >= 3.7.1 and < 3.8** on your computer, in order to make [Pulser](https://pypi.org/project/pulser/) work properly. If you don't have a Python version satisfying these requirements on your computer, use Docker or Anaconda.

### Docker

Open the terminal and:
- Enter the repo: ```cd <path-to-this-repo-folder>```
- Build the docker image: ```docker build -t qaoa-pipeline .```

### Anaconda
TODO
