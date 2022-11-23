
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