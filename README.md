# Reticuler

<p align="center">
<a href="https://pypi.org/project/reticuler/"><img alt="PyPI" src="https://img.shields.io/pypi/v/reticuler"></a>
<a href='https://reticuler.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/reticuler/badge/?version=latest' alt='Documentation Status'/></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Network growth simulations package.

[Documentation](https://reticuler.readthedocs.io/en/latest/)

## Setup

### External dependencies:
[__FreeFEM++__](https://freefem.org/) - PDE solver

### Package installation
Basic usage:
```
pip install .
```

or in the develop mode (overwrites the directory in site-packages with a symbolic link to the repository, hence any changes in code will be automatically reflected):
```
pip install -e .
```

## Usage

During installation two command line scripts are installed:
- *reticulate* - runs the simulation
- *plot_ret* - plots the network based on the *.json* file from the simulation

To use just type in the command line:
`reticulate -h`
or
`plot_ret -h`

Typical network growth simulation:
- output file: *test*,
- growth threshold type: maximum network height,
- growth threshold: 2
```
reticulate -out test --growth_params {\"growth_thresh_type\":1,\"growth_thresh\":2}
```

## How to cite
*Through history to growth dynamics - backward evolution of spatial networks*, S. Żukowski, P. Morawiecki, H. Seybold, P. Szymczak (2022) (accepted)

### References
[1]: [P. Morawiecki, *Problem odwrotny do ewolucji sieci rzecznych* (2016)](http://www.fuw.edu.pl/~piotrek/theses/PMorawiecki.pdf).

[2]: [S. Żukowski, *Związek między geometrią sieci rzecznych a prawami ich wzrostu* (2019)](http://www.fuw.edu.pl/~piotrek/theses/SZukowski.pdf).

[3]: [S. Żukowski, *Backward evolution of river networks* (2021)](http://www.fuw.edu.pl/~piotrek/theses/SZukowski2.pdf).