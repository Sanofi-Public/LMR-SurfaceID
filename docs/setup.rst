Setup
======

* Generally it is recommended to use python version 3.8 for this repo, as not all libraries support python 3.9 yet. Most one-ai workbenches should have conda now which will allow you to easily create a new conda environment with this version.

* In order to run the pipeline and the FastAPI app you will need to configure your PYTHONPATH environment variable to include the src folder in this repo. You can do this by "exporting" the invironment variable in your .bashrc, like this

| ``export PYTHONPATH=$PYTHONPATH:/home/oneai/LMR-SurfaceID``

