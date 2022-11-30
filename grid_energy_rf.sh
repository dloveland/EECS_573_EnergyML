#!/bin/bash
# TODO: Combine all the loops below, didn't realize everything is the same lol
for dataset in "bank" "maternal" "winequality"; do
	EXP_NUM=0
	if [[ $dataset == "winequality" ]]; then
		echo "Num,Max Depth,n_estimators,Max Features,MSE,MAE" > search_results_rf_$dataset.csv
	else
		echo "Num,Max Depth,n_estimators,Max Features,Accuracy,Precision,Recall,F1 Score,Micro F1" > search_results_rf_$dataset.csv
	fi
	for max_d in 3 4 5; do
		for n_estim in 50 100 150; do
			for max_feat in 0.25 0.5 1; do
				echo "P_MAX_D=$max_d" > hyperparams.py
				echo "P_N_ESTIM=$n_estim" >> hyperparams.py
				echo "P_MAX_FEAT=$max_feat" >> hyperparams.py
				echo "EXP_NAME=$EXP_NUM" >> hyperparams.py
				# Just to fix import issues, ignore
				echo "P_LR=None" >> hyperparams.py

				# Do everything for perf, change this to writerow or redirect output to file
				# perf stat -e "power/energy" -x ',' python3 run_ml_model.py --model rf --data $dataset --notune >> energy_rf_$dataset.csv
				python3 run_ml_model.py --model rf --data $dataset --notune >> search_results_rf_$dataset.csv
				EXP_NUM=$((EXP_NUM+1))
			done
		done
	done
done
