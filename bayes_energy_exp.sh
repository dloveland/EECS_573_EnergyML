#!/bin/bash

for model in "rf" "xgb"; do
	printf "Starting $model\n"
	for dataset in "winequality" "bank" "maternal"; do
		echo "Energy,Metric,Counter,Runtime (ms),UNK1,UNK2,UNK3" > bayes_runs/energy_$model\_$dataset.csv
		if [[ $dataset == "winequality" ]]; then
			if [[ $model == "rf" ]]; then
				echo "Tuner,N_iter,Max Depth,Max Features,N Estimators,MSE,MAE,Target" > bayes_runs/results_$model\_$dataset.csv
			else
				echo "Tuner,N_iter,Colsample By Tree,Max Depth,N Estimators,MSE,MAE,Target" > bayes_runs/results_$model\_$dataset.csv
			fi
		else
			if [[ $model == "rf" ]]; then
				echo "Tuner,N_iter,Max Depth,Max Features,N Estimators,Accuracy,Precision,Recall,F1,Micro F1,Target" > bayes_runs/results_$model\_$dataset.csv
			else
				echo "Tuner,N_iter,Colsample By Tree,Max Depth,N Estimators,Accuracy,Precision,Recall,F1,Micro F1,Target" > bayes_runs/results_$model\_$dataset.csv
			fi
		fi
		echo "P_MAX_D=None" > hyperparams.py
		echo "P_N_ESTIM=None" >> hyperparams.py
		echo "P_MAX_FEAT=None" >> hyperparams.py
		echo "EXP_NAME=None" >> hyperparams.py
		echo "P_LR=None" >> hyperparams.py
		echo "P_COL_BT=None" >> hyperparams.py

		printf "Starting $dataset\n"
		for n_iter in {4..27}; do
			# Just to fix import issues, ignore
			echo "P_N_ITER=$n_iter" >> hyperparams.py
			sleep 1
			for tuner in "random" "bayes"; do
				printf "Starting $n_iter $tuner\n"
				sudo perf stat -e "power/energy-pkg/" -x ',' python3 -W ignore run_ml_model.py --model $model --data $dataset --tuner $tuner  &>> bayes_runs/energy_$model\_$dataset.csv

				# Uncomment to do a run to test things
				#python3 run_ml_model.py --model $model --data $dataset --tuner $tuner 
			done
			if [[ $n_iter == 27 ]]; then
				sudo perf stat -e "power/energy-pkg/" -x ',' python3 -W ignore run_ml_model.py --model $model --data $dataset --tuner grid  &>> bayes_runs/energy_$model\_$dataset.csv
			fi
		done
	done
done
printf "Done with $tuner\n"
printf "_________________________________\n"

rm bayes_runs/combined_results_*.csv
python3 stitch_datasets.py --dirname "bayes_runs"
#rm bayes_runs/energy_*.csv
$rm bayes_runs/results_*.csv

mkdir -p all_combs/rf
mkdir -p all_combs/xgb
sudo mv bayes_runs/all_combs/rf_* bayes_runs/all_combs/rf
sudo mv bayes_runs/all_combs/xgb_* bayes_runs/all_combs/xgb
