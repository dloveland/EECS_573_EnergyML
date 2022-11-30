#!/bin/bash
# TODO: Combine all the loops below, didn't realize everything is the same lol
for dataset in "bank" "maternal" "winequality"; do
	EXP_NUM=0
	if [[ $dataset == "winequality" ]]; then
		echo "Num,Max Depth,n_estimators,Max Features,MSE,MAE" > search_results_xgb_$dataset.csv
	else
		echo "Num,Max Depth,n_estimators,Learning Rate,Accuracy,Precision,Recall,F1 Score,Micro F1" > search_results_xgb_$dataset.csv
	fi

	for max_d in 3 4 5; do
		for n_estim in 50 100 150; do
			for lr in 0.1 0.01 0.001; do
				echo "P_MAX_D=$max_d" > hyperparams.py
				echo "P_N_ESTIM=$n_estim" >> hyperparams.py
				echo "P_LR=$lr" >> hyperparams.py
				echo "EXP_NAME=$EXP_NUM" >> hyperparams.py
				# Just to fix import issues, ignore
				echo "P_MAX_FEAT=99999" >> hyperparams.py

				# Do everything for perf, change this to writerow or redirect output to file
				# perf stat -e "power/energy" -x ',' python3 run_ml_model.py --model xgb --data bank --notune >> energy_xgb_bank.csv
				python3 run_ml_model.py --model xgb --data $dataset --notune >> search_results_xgb_$dataset.csv
				EXP_NUM=$((EXP_NUM+1))
			done
		done
	done
done
