"""
FIXME
"""

import pandas as pd
import os
import csv

PROB_TYPE_TO_DS_NAME = {'binary_classification':'bank', 'multiclass_classification':'maternal', 'regression':'winequality'}


""" rewrite to use pd """
def write_row(filename, row, write_rows=False):
    """ Uses csv package to write a given row (or rows) to specified csv file. """
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        if write_rows:
            csv_writer.writerows(row)
        else:
            csv_writer.writerow(row)

def prep_results_files(base_dir="results"):
    if base_dir not in os.listdir():
        os.mkdir(base_dir)
    for problem_dir in ["multiclass_classification", "binary_classification", "regression"]:
        if problem_dir not in os.listdir(base_dir):
            os.mkdir(base_dir +"/"+problem_dir)
        base_model_dir = base_dir + "/" + problem_dir
        for model_class in ["rf", "xgb"]:
            for results_type in ["clf", "results"]:
                if problem_dir == "regression":
                    results_type = "reg"
                dirname = model_class+"_"+results_type
                full_dirname = base_model_dir + "/" +dirname
                if dirname not in os.listdir(base_model_dir):
                    os.mkdir(full_dirname)
        if "rf_results.csv" not in os.listdir(base_model_dir):
            row_header = ["model", "exp_num", "max_depth", "class_weight", "criterion", "n_estimators",
                          "accuracy", "precision", "recall", "f1_score", "micro_avg_f1_score"]
            if problem_dir == "regression":
                row_header = ["model", "exp_num", "max_depth", "criterion", "n_estimators", "mse", "mae"]
            write_row(base_model_dir + "/rf_results.csv", row_header)
        if "xgb_results.csv" not in os.listdir(base_model_dir):
            row_header = ["model", "exp_num", "max_depth", "learning_rate", "gamma", "n_estimators",
                          "accuracy", "precision", "recall", "f1_score", "micro_avg_f1_score"]
            if problem_dir == "regression":
                row_header = row_header[:-5]+["mse", "mae"]
            write_row(base_model_dir + "/xgb_results.csv", row_header)

def print_best_results(filename, model_type, prob_type):
    results = pd.read_csv(filename)
    if prob_type == "regression":
        argmin_mse = results["mse"].idxmin()
        argmin_mae = results["mae"].idxmin()

        print("Best results for {0} model ({1} dataset -- {2})".format(model_type, PROB_TYPE_TO_DS_NAME[prob_type], prob_type))
        print("MSE: exp_num = {0} ({1})".format(results["exp_num"][argmin_mse], results["mse"][argmin_mse]))
        print("MAE: exp_num = {0} ({1})".format(results["exp_num"][argmin_mae], results["mse"][argmin_mae]))
        print()

    else:
        argmax_precision = results["precision"].idxmax()
        argmax_recall = results["recall"].idxmax()
        argmax_f1_score = results["f1_score"].idxmax()
        argmax_micro_avg_f1 = results["micro_avg_f1_score"].idxmax()

        print("Best results for {0} model ({1} dataset -- {2})".format(model_type, PROB_TYPE_TO_DS_NAME[prob_type], prob_type))
        print("Precision: exp_num = {0} ({1})".format(results["exp_num"][argmax_precision], results["precision"][argmax_precision]))
        print("Recall: exp_num = {0} ({1})".format(results["exp_num"][argmax_recall], results["recall"][argmax_recall]))
        print("F1 Score: exp_num = {0} ({1})".format(results["exp_num"][argmax_f1_score], results["f1_score"][argmax_f1_score]))
        print("Micro AVG F1 Score: exp_num = {0} ({1})".format(results["exp_num"][argmax_micro_avg_f1],results["micro_avg_f1_score"][argmax_micro_avg_f1]))
        print()

if __name__ == "__main__":
    prep_results_files()
