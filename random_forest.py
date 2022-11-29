"""
FIXME
"""

import numpy as np
import pandas as pd
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

# Make this a flag and pass as input later
OPT_STYLE = "Bayes"


def validate_rf(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    if prob_type == "regression":
        validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir)
    else:
        validate_rf_classifier(x_train,  y_train, x_test, y_test, prob_type, base_dir)

def validate_rf_classifier(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    exp_num = 0
    '''
    Code for manual grid search
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
    '''
    params = {
        'criterion':["gini", "entropy", "log_loss"],
        'max_depth':[3, 4],
        'n_estimators':[50, 200, 400],
        'class_weight':['balanced',None],
    }

    # Maybe change this to weighted or samples based f1
    scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_micro','recall_micro','f1_macro', 'f1_micro']
    refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

    hp_tuner = None
    if OPT_STYLE == 'Bayes':
        print("Performing Bayesian Search")
        hp_tuner = BayesSearchCV(
            RandomForestClassifier(random_state=573,n_jobs=-1),
            search_spaces=params,
            cv=5,
            #scoring=scoring_options,
            scoring='f1',
            refit=refit_target,
            )
    else:
        print("Performing Grid Search")
        hp_tuner = GridSearchCV(
            RandomForestClassifier(random_state=573,n_jobs=-1),
            param_grid=params,
            cv=5,
            scoring=scoring_options,
            refit=refit_target,
            )
    hp_results = hp_tuner.fit(x_train, y_train)
    print(f"Best config found was {hp_results.best_params_} with the best {refit_target} of {hp_results.best_score_}.\n")

    rf_clf = hp_results.best_estimator_
    preds = rf_clf.predict(x_test)
    dump(rf_clf, base_dir+"/rf_clf/rf_best_" + refit_target + ".joblib")
    report = classification_report(y_test, preds, output_dict=True)
    confusion_mat = confusion_matrix(y_test, preds)
    np.savez(base_dir + "/rf_results/rf_best_" + refit_target  + ".npz", confusion_mat)
    micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)
    exp_results = {"Accuracy":accuracy,
            "Precision":report["macro avg"]["precision"],
            "Recall":report["macro avg"]["recall"],
            "F1":report["macro avg"]["f1-score"],
            "Micro F1":micro_avg_f1}
    print("Testing results (Macro Avg for Multilabel) were:")
    for k,v in exp_results.items():
        print(f"{k}:{v}")

    pd.DataFrame(hp_results.cv_results_).to_csv(base_dir+"/"+filename)



def validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="rf_results.csv"):
    '''
    Manual Grid Search, added a GridSearchCV based implementation for consistency with Bayes, can change back later;

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
    '''
    params = {
        'criterion':["squared_error", "absolute_error", "poisson"],
        'max_depth':[3, 4],
        'n_estimators':[50, 200, 400],
    }
    refit_target = 'neg_mean_squared_error'
    hp_tuner = None
    if OPT_STYLE == 'Bayes':
        print("Performing Bayesian Search")
        hp_tuner = BayesSearchCV(
            RandomForestRegressor(random_state=573,n_jobs=-1),
            search_spaces=params,
            cv=5,
            #scoring=scoring_options,
            scoring='f1',
            refit=refit_target,
            )
    else:
        print("Performing Grid Search")
        hp_tuner = GridSearchCV(
            RandomForestClassifier(random_state=573,n_jobs=-1),
            param_grid=params,
            cv=5,
            scoring=scoring_options,
            refit=refit_target,
            )
    hp_results = hp_tuner.fit(x_train, y_train)
    print(f"Best config found was {hp_results.best_params_} with the best {refit_target} of {hp_results.best_score_}.\n")

    rf_reg = hp_results.best_estimator_
    preds = rf_reg.predict(x_test)
    dump(rf_reg, base_dir+"/rf_reg/rf_best_"+refit_target+".joblib")
    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    mae = sklearn.metrics.mean_absolute_error(y_test, preds)
    exp_results = {"MSE":mse, "MAE":mae}
    print("Testing results were:")
    for k,v in exp_results.items():
        print(f"{k}:{v}")

    pd.DataFrame(hp_results.cv_results_).to_csv(base_dir+"/"+filename)
