# Team Index
Prototype code for the TeamIndex idea, a scalable approach for accelerating high-dimensional range queries at the very large scale.
The core idea is to employ an ensemble of indices over (potentially disjoint) subsets of table attributes, and relying on efficient intersection for identifying qualifying tuple IDs.

This project showcases the principle idea and the linear scaling (w.r.t. table size/dimensionality) of a Team-based approach to secondary indexing.


## Recreating Figures

To recreate figures, follow the steps below to install required python packages for plotting inside a fresh virtual environment:

    python3 -m venv venv
    source venv/bin/activate
  
    pip install matplotlib seaborn pandas numpy pyarrow
  
    ./create_figures.py

The script then renders the figures using our previous measurements stored in `./data`.

## Prototype Code

Note: code and scripts developed and tested on archlinux.

To run code for creating and evaluating indices using a small and simple uniform dataset:

    source venv/bin/activate
    pip install ./code/

    cd minimal_example
    python -i minimal_example/run_example.py


The program code can be found in the subfolder "code".