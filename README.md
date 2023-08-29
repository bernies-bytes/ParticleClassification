# ParticleClassification

Python project to classify signal and background particles. 

Available parts of the pipeline:

- Data:
    - Creating synthetic data for the classification problem (`synth_data/create_data.py`).
    - Or reading in your own data (TODO).
    - Over- and under-sampling and a combination of both (`sampling.py`). 

Up next:
- Feature selection, feature importance.
- Feature engineeing, feature trafo. 
- Model setup - hyper parameter tuning.
- Training, Testing, Cross validation.
- ...


## How to run

To run, execute: `master_script.py classification.conf`

See the config file `classification.conf` for what parts of the pipeline to execute and available settings. `utils/config_setup.py` takes care of reading the config file and setting up directories, logger etc. for the run of the pipeline.
