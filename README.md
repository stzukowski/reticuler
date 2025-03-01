# Reticuler

<p align="center">
<a href="https://pypi.org/project/reticuler/"><img alt="PyPI" src="https://img.shields.io/pypi/v/reticuler"></a>
<a href='https://reticuler.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/reticuler/badge/?version=latest' alt='Documentation Status'/></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Python package to simulate the growth of spatial transport networks in nature.

[Documentation](https://reticuler.readthedocs.io/en/latest/)

## Setup

### External dependencies:
[__FreeFEM++__](https://freefem.org/) - PDE solver

### Package installation
```
pip install reticuler
```

## Usage

Four command line scripts are installed during the installation:
   - *reticulate* - runs the simulation
   - *reticulate_back* - runs the the Backward Evolution Algorithm
   - *clip_ret* - clips the network to one of the growth thresholds: maximum forward evolution step, length, height, evolution time, or BEA step
   - *plot_ret* - plots the network based on the *.json* file from the simulation
   - *script_ret* - prepares a FreeFEM++ script based on the *.json* file

To use just type in the command line:
`reticulate -h`

Typical network growth simulation:
```
reticulate -out test --growth_params {\"growth_thresh_type\":1,\"growth_thresh\":2}
```
- output file: *test*,
- growth threshold type: maximum network height,
- growth threshold: 2

## How to cite
[1] [*Through history to growth dynamics: backward evolution of spatial networks*](https://doi.org/10.1038/s41598-022-24656-x), S. Żukowski, P. Morawiecki, H. Seybold, P. Szymczak, Sci. Rep. 12, 20407 (2022).
<!--- [Materials](https://github.com/stzukowski/reticuler/tree/main/archive/papers/2022SciRep) --->
[2] [*Breakthrough-induced loop formation in evolving transport networks*](https://doi.org/10.1073/pnas.2401200121), S. Żukowski, A. J. M. Cornelissen, F. Osselin, S. Douady, P. Szymczak, PNAS 121 (29), e2401200121 (2024).
[Materials](https://github.com/stzukowski/reticuler/tree/main/archive/2024PNAS)

**References:**
The thin-finger growth algorithm used in this package was based on an earlier code described in the paper
[*Bifurcation dynamics of natural drainage networks*](https://doi.org/10.1098/rsta.2012.0365) (A. Petroff, O. Devauchelle, H. Seybold, and D. H. Rothman. Philos. Trans. Royal Soc. A 371, 20120365, 2013)
