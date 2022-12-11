# Energy Analysis of HP Tuning for RFs and GBTs

## Dependencies
All the Python dependencies can be installed by:\
`pip install scikit-optimize scikit-learn xgboost numpy pandas` 

Scripts also require Linux Perf Tool to be installed. In case you are unable to do so, remove the usage of perf on top of the Python package calls.

## Running the Scripts
All the scripts should be run from the root directory.

`./scripts/percomb_energy_exp.sh` runs all experiments for the intra-model and inter-model experiments. \
`./scripts/bayes_energy_exp.sh` runs all experiments for the tuning experiments. \
