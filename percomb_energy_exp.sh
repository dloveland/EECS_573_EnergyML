#!/bin/bash
# TODO: Combine all the loops below, didn't realize everything is the same lol

printf "Starting RF\n"
for dataset in "bank" "maternal" "winequality"; do
	printf "Starting RF on $dataset\n"
	EXP_NUM=0
	echo "Energy,Metric,Counter,Runtime (ms),UNK1,UNK2,UNK3" > percomb_runs/energy_rf_$dataset.csv
	if [[ $dataset == "winequality" ]]; then
		echo "Num,Max Depth,n_estimators,Max Features,MSE,MAE" > percomb_runs/results_rf_$dataset.csv
	else
		echo "Num,Max Depth,n_estimators,Max Features,Accuracy,Precision,Recall,F1 Score,Micro F1" > percomb_runs/results_rf_$dataset.csv
	fi

	for max_d in 3 4 5; do
		for n_estim in 50 100 150; do
			for max_feat in 0.2 0.5 0.8; do
				echo "P_MAX_D=$max_d" > hyperparams.py
				echo "P_N_ESTIM=$n_estim" >> hyperparams.py
				echo "P_MAX_FEAT=$max_feat" >> hyperparams.py
				echo "EXP_NAME=$EXP_NUM" >> hyperparams.py
				# Just to fix import issues, ignore
				echo "P_LR=None" >> hyperparams.py
				echo "P_COL_BT=None" >> hyperparams.py

				# Do everything for perf, change this to writerow or redirect output to file
				sudo perf stat -e "power/energy-pkg/" -x ',' python3 -W ignore run_ml_model.py --model rf --data $dataset --notune  &>> percomb_runs/energy_rf_$dataset.csv

				# Uncomment to do a testing perfless run
				# cat percomb_runs/results_rf_$dataset.csv
				# python3 run_ml_model.py --model rf --data $dataset --notune
				EXP_NUM=$((EXP_NUM+1))
				printf "Finished {$max_d,$n_estim,$max_feat}\n"
			done
		done
	done
done
echo "Done with RF\n"
echo "_________________________________\n"

for dataset in "bank" "maternal" "winequality"; do
	printf "Starting XGB on $dataset\n"
	EXP_NUM=0
	echo "Energy,Metric,Counter,Runtime (ms),UNK1,UNK2,UNK3" > percomb_runs/energy_xgb_$dataset.csv
	if [[ $dataset == "winequality" ]]; then
		echo "Num,Max Depth,N_estimators,Colsample By Tree,Learning Rate,MSE,MAE" > percomb_runs/results_xgb_$dataset.csv
	else
		echo "Num,Max Depth,N_estimators,Colsample By Tree,Learning Rate,Accuracy,Precision,Recall,F1 Score,Micro F1" > percomb_runs/results_xgb_$dataset.csv
	fi

	for max_d in 3 4 5; do
		for n_estim in 50 100 150; do
			for col_bt in 0.2 0.5 0.8; do
				for lr in 0.1 0.01 0.001; do
					echo "P_MAX_D=$max_d" > hyperparams.py
					echo "P_N_ESTIM=$n_estim" >> hyperparams.py
					echo "P_LR=$lr" >> hyperparams.py
					echo "P_COL_BT=$col_bt" >> hyperparams.py
					echo "EXP_NAME=$EXP_NUM" >> hyperparams.py
					# Just to fix import issues, ignore
					echo "P_MAX_FEAT=None" >> hyperparams.py

					# Do everything for perf, change this to writerow or redirect output to file
					sudo perf stat -e "power/energy-pkg/" -x ',' python3 -W ignore run_ml_model.py --model xgb --data $dataset --notune  &>> percomb_runs/energy_xgb_$dataset.csv

					# Uncomment to do a testing perfless run
					# cat percomb_runs/results_xgb_$dataset.csv
					# python3 run_ml_model.py --model xgb --data $dataset --notune
					EXP_NUM=$((EXP_NUM+1))
					printf "Finished {$max_d,$n_estim,$col_bt,$lr}\n" 
				done
			done
		done
	done
done

rm percomb_runs/combined_results_*.csv
python3 stitch_datasets.py --dirname "percomb_runs"

rm percomb_runs/energy_*.csv
rm percomb_runs/results_*.csv

printf "Done with XGB\n"
printf "_________________________________\n"
