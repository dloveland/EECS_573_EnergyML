# Energy Analysis of HP Tuning for RFs and GBTs

## Dependencies
All the Python dependencies can be installed by:\
`pip install scikit-optimize scikit-learn xgboost numpy pandas` 

Scripts also require Linux Perf Tool to be installed. In case you are unable to do so, remove the usage of perf on top of the Python package calls.

## Running the Scripts
All the scripts should be run from the root directory.
All the scripts should be run from the root directory.


`./scripts/percomb_energy_exp.sh` runs all experiments for the intra-model and inter-model experiments. \
`./scripts/percomb_energy_exp.sh` runs all experiments for the Intra-model and Inter-model experiments. All results are saved in `percomb_runs`
`./scripts/bayes_energy_exp.sh` runs all experiments for the tuning experiments. \

`./scripts/bayes_energy_exp.sh` runs all experiments for the Tuning experiments. All results are saved in `bayes_runs`

`./scripts/tuning_runs.sh` is a legacy script for running the Tuning experiments without seeding or varying the max iterations. All results are saved in `tuning_runs`
