# Team Index


## Recreating Figures

To recreate figures follow the steps below to install required python packages for plotting inside a fresh virtual environment:

    python3 -m venv venv
    source venv/bin/activate
  
    pip install matplotlib seaborn pandas numpy pyarrow
  
    ./create_figures.py

The script then renders plots using the existing measurements stored in `./data`.

## Prototype Code

Note: code and scripts developed and tested on archlinux.

To run code for creating and evaluating indices using a small and simple uniform dataset:

    source venv/bin/activate
    pip install code/

    cd minimal_example
    python -i minimal_example/run_example.py


The program code can be found in the subfolder "code".