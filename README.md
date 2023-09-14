# ParticleClassification

Python project to classify signal and background particle events in a detector (under construction). 

Available parts of the pipeline:

- Data:
    - Creating synthetic data for the classification problem (`synth_data/create_data.py`).
    - Or reading in your own data (TODO).
    - Over- and under-sampling and a combination of both (`sampling.py`). 
    - Train-test splitting of the data set (`train_test_splitting.py`).
    - Hyper-parameter tuning (`hyper_parameter_tuning.py`). So far, Baysian optimisation with `bayes_opt` is possible. 

To-do:
- TPE (tree parzen estimator) hyper parameter tuning. 
- Feature selection, feature importance.
- Feature engineeing, feature trafo. 

Backlog:
- Model setup - hyper parameter tuning.
- Training, Testing, Cross validation.
- ...


## How to run

To run, execute: `master_script.py classification.conf`

See the config file `classification.conf` for what parts of the pipeline to execute and available settings. `utils/config_setup.py` takes care of reading the config file and setting up directories, logger etc. for the run of the pipeline.
