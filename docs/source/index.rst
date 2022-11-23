
=====================================
Reticuler Documentation
=====================================

.. toctree::
   :hidden:
   :maxdepth: 1
   
   modules
   
How to install
''''''''''''''

Basic usage:

.. prompt:: bash $ auto

   pip install .

or in the develop mode (overwrites the directory in site-packages with a symbolic link to the repository, hence any changes in code will be automatically reflected):

.. prompt:: bash $ auto

   pip install -e .
   
   
Usage
'''''

During installation two command line scripts are installed:
   - *reticulate* - runs the simulation
   - *plot_ret* - plots the network based on the *.json* file from the simulation

To use just type in the command line:

.. prompt:: bash $ auto

   reticulate -h
   
or

.. prompt:: bash $ auto

   plot_ret -h

Typical network growth simulation:
   - output file: *test*,
   - growth threshold type: maximum network height,
   - growth threshold: 2

.. prompt:: bash $ auto

   reticulate -out test --growth_params {\"growth_thresh_type\":1,\"growth_thresh\":2}