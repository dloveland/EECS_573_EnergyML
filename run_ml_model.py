"""
FIXME
"""
import argparse

import pandas as pd
from datetime import datetime
from file_utils import *
from preprocessing import read_and_split_data

from random_forest import *
from xgb import *

DS_NAME_TO_PROB_TYPE = {'bank': 'binary_classification', 'maternal':'multiclass_classification', 'winequality':'regression'}

def run_validation_exp(dataset_name, model, params, base_dir="results"):
    x_tr, y_tr, x_vl, y_vl, x_ts, y_ts = read_and_split_data(dataset_name)
    prep_results_files(base_dir)
    prob_type = DS_NAME_TO_PROB_TYPE[dataset_name]
    base_dir = base_dir + "/" + prob_type

    if model == "rf":
        validate_rf(x_tr, y_tr, x_vl, y_vl, prob_type, params, base_dir=base_dir)
    if model == "xgb":
        validate_xgb(x_tr, y_tr, x_vl, y_vl, prob_type, params, base_dir=base_dir)

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
    parser = argparse.ArgumentParser();
    parse.add_argument("--model", type=str, default="rf", help = "Train one of Random Forests or XGB based predictors");
    parse.add_argument("--dataset", type=str, default="bank", help = "Train model on either bank, maternal or winequality datasets");

    # General params
    parse.add_argument("--maxdepth", type=int, default=3, help = "Set Max Depth of Decision Trees");
    parse.add_argument("--nestim", type=int, default=50, help = "Set number of estimators for Decision Trees");

    # XGB specific
    parse.add_argument("--lr", type=float, default=0.1, help = "Set learning rate to train XGB Decision Trees");
    parse.add_argument("--gamma", type=int, default=0, help = "Set number Gamma to train XGB Decision Trees");

    # RF specific
    parse.add_argument("--criterion", type=str, default="gini", help = "Criterion function to train RF Decision Trees");
    parse.add_argument("--weights", type=str, default=None, help = "Criterion function to train RF Decision Trees");

    args = parse.parse_args();
    start = datetime.now()
    params = {"max_depth":args.maxdepth, "n_estimators":args.nestim, "lr":args.lr, "gamma":args.gamma,
            "criterion":args.criterion, "weights":args.weights}
    #for dataset_name in ['bank', 'maternal', 'winequality']:
    #    for model in ["rf", "xgb"]:

    print(f"Running Experiment using {args.model} on dataset {args.dataset}.")
    print("Printing params dict for sanity check.")
    print(params)
    run_validation_exp(args.dataset, args.model, params=params)
    end = datetime.now()
    print("Total time: ", end - start)

if __name__ == "__main__":
    main()
