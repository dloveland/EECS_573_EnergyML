#!/bin/bash

for model in "rf" "xgb"; do
	for dataset in "bank" "maternal" "winequality"; do
		echo "Energy,Metric,Counter,Runtime (ms),UNK1,UNK2,UNK3" > tuning_runs/energy_$model\_$dataset.csv
		if [[ $dataset == "winequality" ]]; then
			if [[ $model == "rf" ]]; then
				echo "Tuner,Max Depth,Max Features,N Estimators,MSE,MAE,Target" > tuning_runs/results_$model\_$dataset.csv
			else
				echo "Tuner,Colsample By Tree,Max Depth,N Estimators,MSE,MAE,Target" > tuning_runs/results_$model\_$dataset.csv
			fi
		else
			if [[ $model == "rf" ]]; then
				echo "Tuner,Max Depth,Max Features,N Estimators,Accuracy,Precision,Recall,F1,Micro F1,Target" > tuning_runs/results_$model\_$dataset.csv
			else
				echo "Tuner,Colsample By Tree,Max Depth,N Estimators,Accuracy,Precision,Recall,F1,Micro F1,Target" > tuning_runs/results_$model\_$dataset.csv
			fi
		fi

		for tuner in "grid" "bayes"; do
			printf "Starting $model $dataset $tuner\n"
			sudo perf stat -e "power/energy-pkg/" -x ',' python3 -W ignore run_ml_model.py --model $model --data $dataset --tuner $tuner  &>> tuning_runs/energy_$model\_$dataset.csv

			# Uncomment to do a run to test things
			#python3 run_ml_model.py --model $model --data $dataset --tuner $tuner 
		done
	done
done
printf "Done with $tuner\n"
printf "_________________________________\n"

rm tuning_runs/combined_results_*.csv
python3 stitch_datasets.py --dirname "tuning_runs"
rm tuning_runs/energy_*.csv
rm tuning_runs/results_*.csv
