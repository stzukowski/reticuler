
=====================================
Reticuler Documentation
=====================================

.. toctree::
   :hidden:
   :maxdepth: 1
   
   modules
   
Reticuler is a Python package to simulate the growth of spatial networks in nature.
   
How to install
''''''''''''''

.. prompt:: bash $ auto

   pip install reticuler

   
   
Usage
'''''

During the installation four command line scripts are installed:
   - *reticulate* - runs the simulation
   - *reticulate_back* - runs the the Backward Evolution Algorithm
   - *clip_ret* - clips the network to one of the growth thresholds (maximum forward evolution step, length, height, evolution time, or BEA step)
   - *plot_ret* - plots the network based on the *.json* file from the simulation

To use just type in the command line:

.. prompt:: bash $ auto

   reticulate -h
   
Typical network growth simulation:
   - output file: *test*,
   - growth threshold type: maximum network height,
   - growth threshold: 2

.. prompt:: bash $ auto

   reticulate -out test --growth_params {\"growth_thresh_type\":1,\"growth_thresh\":2}