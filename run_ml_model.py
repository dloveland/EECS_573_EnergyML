"""
FIXME
"""

import argparse
import pandas as pd
from datetime import datetime
from file_utils import *
from preprocessing import read_and_split_data

from random_forest import *
from xgb import validate_xgb

DS_NAME_TO_PROB_TYPE = {'bank': 'binary_classification', 'maternal':'multiclass_classification', 'winequality':'regression'}

def run_validation_exp(dataset_name, model, tuner=None, notune=False, base_dir="results"):
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = read_and_split_data(dataset_name)
    prep_results_files(base_dir)
    prob_type = DS_NAME_TO_PROB_TYPE[dataset_name]
    base_dir = base_dir + "/" + prob_type

    if model == "rf":
        validate_rf(x_tr, y_tr, x_vl, y_vl, prob_type, tuner, notune, base_dir=base_dir)
    if model == "xgb":
        validate_xgb(x_tr, y_tr, x_vl, y_vl, prob_type, tuner, notune, base_dir=base_dir)

def get_test_results(dataset_name, model, exp_num, base_dir="results"):
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = read_and_split_data(dataset_name)
    base_dir += "/" + DS_NAME_TO_PROB_TYPE[dataset_name]

    results = base_dir + "/" + model + "_results.csv"
    results_df = pd.read_csv(results)

    if model == "xgb":
        y_ts = np.array(y_ts, dtype="int64")

    filepath = base_dir + "/" + model + "_clf/" + model + "_exp_num" + str(exp_num) + ".joblib"
    clf = load(filepath)
    preds = clf.predict(x_ts)
    report = classification_report(y_ts, preds, output_dict=True)
    confusion_mat = confusion_matrix(y_ts, preds)
    micro_avg_f1 = sklearn.metrics.f1_score(y_ts, preds, average="micro")

    print("Test Results for", model, "Exp", exp_num)
    print("Precision:", report["1"]["precision"])
    print("Recall:", report["1"]["recall"])
    print("F1 Score:", report["1"]["f1-score"])
    print("Micro Avg F1 Score:", micro_avg_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help = "Train one of Random Forests or XGB based predictors")
    parser.add_argument("--dataset", type=str, default="all", help = "Train model on either bank, maternal or winequality datasets")
    parser.add_argument("--tuner", type=str, default="grid", help = "Tuning Style: grid does GridSearchCV and bayes does bayesian optimization for the same")
    parser.add_argument('--notune', action='store_true')
    args = parser.parse_args()

    if args.model=='all' or args.dataset=='all':
        for dataset_name in ['bank', 'maternal', 'winequality']:
            for model in ["rf", "xgb"]:
                print(f"Running experiment for {model} on {dataset_name}")
                run_validation_exp(dataset_name, model, tuner=args.tuner)
                print('─' * 40)
    else:
        model = args.model
        dataset_name = args.dataset
        run_validation_exp(dataset_name, model, tuner=args.tuner, notune=args.notune)


if __name__ == "__main__":
    main()
