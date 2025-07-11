# TeamIndex

## Project Structure

This is a C++ project with a python frontend that also contains various utilities, especially for creating indices and diagnosis. The project can be fully and conveniently installed using python pip. 

Running the index creation requires the python frontend to provide necessary meta information, but there is an alternative (and separately compilable) standalone executable for running queries on existing indices. However, this standalone requires a json query specification that contains specifications for all I/O requests. These are cumbersome to write manually, so getting the python package running is advisable.



### Important Files

- Create index: 
    - code/cpp/include/create/create.hpp
    - There are other files that implement similar functionality, but they are not finished
- Run queries:
    - code/cpp/src/runtime/runtime.cpp (TeamIndexExecutor::run)
    - Wrapping standalone: code/cpp/src/runtime/standalone_runtime.cpp
- TeamIndex Python class that manages meta-data and request generation:
    - code/python/TeamIndex/evaluation.py


## Install Code
Linux only, not tested for systems other than ubuntu and archlinux.

First:
- Install mandatory dependency `liburing-dev` (ubuntu)
- Optionally, install `pybind11`, `libzstd-dev` in your system
    - If you do not, we download them later either in a virtualenv (with pip) or within the project folder (using CMake's FetchContent)



Then, clone the git project and change directory into the project folder. From within the folder:
- `virtualenv --python=3.12 build_env`
- `source build_env/bin/activate`
- `pip install scikit_build_core setuptools_scm pyyaml numpy pandas pyarrow pybind11`

Optionally:
- Enable additional compressions:
    - `ENABLE_FASTPFOR=true`
        - Note: We use tag `v0.3.0`


Note:
- TaskFlow is a header-only external dependency we copied into a subfolder. We use GIT_TAG v3.9.0.
    - There is a trivial changes to the TaskView class: We added a getter for the data() member of an underlying task (to implement custom time tracking using Observers).


## Build

With pip:
- Default, e.g. for benchmarks:
    - `pip install .`
- For development: `python -m pip install -vvv . --no-clean --no-build-isolation --config-settings=cmake.build-type="Debug"`
    - Note: requires installing python build-time dependencies beforehand

## Run

- Source the virtualenv or otherwise make sure the runtime python dependencies are ensured
- Run the scripts


