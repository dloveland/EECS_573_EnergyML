"""
FIXME
"""

import numpy as np
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def validate_rf(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    if prob_type == "regression":
        validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir)
    else:
        validate_rf_classifier(x_train,  y_train, x_test, y_test, prob_type, base_dir)

def validate_rf_classifier(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    exp_num = 0
    for max_depth in [3, 4]:
        for class_weight in ["balanced", None]:
            for criterion in ["gini", "entropy", "log_loss"]:
                for n_estimators in range(50, 400, 50):
                    rf_clf = RandomForestClassifier(max_depth=max_depth, class_weight=class_weight,
                                                    criterion=criterion, n_estimators=n_estimators)
                    rf_clf.fit(x_train, y_train)

                    dump(rf_clf, base_dir+"/rf_clf/rf_exp_num" + str(exp_num) + ".joblib")
                    preds = rf_clf.predict(x_test)
                    report = classification_report(y_test, preds, output_dict=True)
                    confusion_mat = confusion_matrix(y_test, preds)
                    np.savez(base_dir + "/rf_results/rf_exp_num" + str(exp_num) + ".npz", confusion_mat)
                    micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
                    accuracy = sklearn.metrics.accuracy_score(y_test, preds)
                    exp_results = [accuracy, report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], micro_avg_f1]
                    hp = ["rf", exp_num, max_depth, class_weight, criterion, n_estimators]
                    write_row(base_dir+"/"+filename, np.array(hp + exp_results))
                    print("Done with experiment " + str(exp_num))
                    exp_num += 1
                    break # FIXME
                break # FIXME
            break # FIXME


    print_best_results(base_dir+"/"+filename, "RF", prob_type)


def validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    exp_num = 0
    for max_depth in [3, 4]:
        for criterion in ["squared_error", "absolute_error", "poisson"]:
            for n_estimators in range(50, 400, 50):
                rf_reg = RandomForestRegressor(max_depth=max_depth, criterion=criterion, n_estimators=n_estimators)
                rf_reg.fit(x_train, y_train)
                dump(rf_reg, base_dir+"/rf_reg/rf_exp_num" + str(exp_num) + ".joblib")
                preds = rf_reg.predict(x_test)
                mse = sklearn.metrics.mean_squared_error(y_test, preds)
                mae = sklearn.metrics.mean_absolute_error(y_test, preds)
                exp_results = [mse, mae]
                hp = ["xgb", exp_num, max_depth, criterion, n_estimators]
                write_row(base_dir+"/"+filename, np.array(hp + exp_results))
                print("Done with experiment " + str(exp_num))
                exp_num += 1
                break # FIXME
            break # FIXME
    print_best_results(base_dir+"/"+filename, "RF", prob_type)
