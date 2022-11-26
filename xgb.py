"""
FIXME
"""

import numpy as np
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

import xgboost as xgb

def validate_xgb(x_train, y_train, x_test, y_test, prob_type, params, base_dir, filename="xgb_results.csv"):
    if prob_type == "regression":
        validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, params, base_dir)
    else:
        validate_xgb_classifier(x_train,  y_train, x_test, y_test, prob_type, params, base_dir)

def validate_xgb_classifier(x_train, y_train, x_test, y_test, prob_type, params, base_dir, filename="xgb_results.csv"):
    exp_num = 0
    y_test = np.array(y_test, dtype="int64")
    max_depth = params['max_depth']
    learning_rate = params['learning_rate']
    gamma = params['gamma']
    n_estimators = params['n_estimators']

    if "binary" in prob_type:
        objective = "binary:logistic"
    else:
        objective = "multi:softmax"
    xgb_clf = xgb.XGBClassifier(objective=objective,eval_metric="logloss",use_label_encoder=False,
                                max_depth=max_depth, learning_rate=learning_rate, gamma=gamma, n_estimators=n_estimators)
    xgb_clf.fit(x_train, y_train)
    dump(xgb_clf, base_dir+"/xgb_clf/xgb_exp_num" + str(exp_num) + ".joblib")
    preds = xgb_clf.predict(x_test)
    report = classification_report(y_test, preds, output_dict=True)
    confusion_mat = confusion_matrix(y_test, preds)
    np.savez(base_dir + "/xgb_results/xgb_exp_num" + str(exp_num) + ".npz", confusion_mat)
    micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)
    exp_results = [accuracy, report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], micro_avg_f1]
    hp = ["xgb", exp_num, max_depth, learning_rate, gamma, n_estimators]
    write_row(base_dir+"/"+filename, np.array(hp + exp_results))
    print("Done with experiment " + str(exp_num))

    print("Parameters used were")
    print(f"Max depth: {max_depth}, Learning Rate: {learning_rate}, Gamma: {gamma}, Num Estimators: {n_estimators}\n")
    print(f'Results were:')
    print(f'Accuracy: {accuracy}, Precision: {report["1"]["precision"]}, Recall: {report["1"]["recall"]}
            , F1-score: {report["1"]["f1-score"]}, Micro-avg-f1: {micro_avg_f1}')
    
    print_best_results(base_dir+"/"+filename, "XGBoost", prob_type)

def validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, params, base_dir, filename="xgb_results.csv"):
    # TODO: Fix exp_num or remove it
    exp_num = 0
    y_test = np.array(y_test, dtype="int64")
    max_depth = params['max_depth']
    learning_rate = params['learning_rate']
    gamma = params['gamma']
    n_estimators = params['n_estimators']

    xgb_reg = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, gamma=gamma, n_estimators=n_estimators)
    xgb_reg.fit(x_train, y_train)
    dump(xgb_reg, base_dir+"/xgb_reg/xgb_exp_num" + str(exp_num) + ".joblib")
    preds = xgb_reg.predict(x_test)
    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    mae = sklearn.metrics.mean_absolute_error(y_test, preds)
    exp_results = [mse, mae]
    hp = ["xgb", exp_num, max_depth, learning_rate, gamma, n_estimators]
    write_row(base_dir+"/"+filename, np.array(hp + exp_results))
    print("Done with experiment " + str(exp_num))
    exp_num += 1

    print("Parameters used were")
    print(f"Max depth: {max_depth}, Learning Rate: {learning_rate}, Gamma: {gamma}, Num Estimators: {n_estimators}\n")
    print("Results Were")
    print(f"MSE: {mse}, MAE: {mae}")
    print_best_results(base_dir+"/"+filename, "XGBoost", prob_type)