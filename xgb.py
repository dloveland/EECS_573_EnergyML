import numpy as np
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from skopt import BayesSearchCV

# Make this a flag and pass as input later
OPT_STYLE = "Bayes"

def validate_xgb(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="xgb_results.csv"):
    if prob_type == "regression":
        validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir)
    else:
        validate_xgb_classifier(x_train,  y_train, x_test, y_test, prob_type, base_dir)

def validate_xgb_classifier(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="xgb_results.csv"):
    '''
    exp_num = 0
    y_test = np.array(y_test, dtype="int64")
    for max_depth in [3, 4]:
        for learning_rate in [0.1, 0.01, 0.001]:
            for gamma in list(range(0, 25, 5)) + [100, 1000]:
                for n_estimators in range(50, 400, 50):
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
                    exp_num += 1
                    break # FIXME
                break # FIXME
            break # FIXME

    print_best_results(base_dir+"/"+filename, "XGBoost", prob_type)
    '''
    y_test = np.array(y_test, dtype="int64")
    params = {
        'max_depth':[3, 4],
        'n_estimators':[50, 200, 400],
        'learning_rate':[0.1, 0.01, 0.001],
        'gamma':[0,10,25,100, 1000],
    }
    obj = 'binary:logistic' if 'binary' in prob_type else 'multi_softmax'
    scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_micro','recall_micro','f1_macro', 'f1_micro']
    refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

    if OPT_STYLE == 'Bayes':
        print("Performing Bayesian Search")
        hp_tuner = BayesSearchCV(
               xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
               search_spaces=params,
               cv=5,
               scoring=refit_target,
               refit=refit_target,
        )
    else:
        print("Performing Grid Search")
        hp_tuner = GridSearchCV(
               xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
               param_grid=params,
               cv=5,
               scoring=scoring_options,
               refit=refit_target,
        )
    hp_results = hp_tuner.fit(x_train, y_train)
    print(f"Best config found was {hp_results.best_params_} with the best {refit_target} of {hp_results.best_score_}.\n")

    # Mostly the old code
    xgb_clf = hp_results.best_estimator_
    preds = xgb_clf.predict(x_test)
    dump(xgb_clf, base_dir+"/xgb_results/xgb_best_" + refit_target + ".joblib")
    report = classification_report(y_test, preds, output_dict=True)
    confusion_mat = confusion_matrix(y_test, preds)
    np.savez(base_dir + "/xgb_results/xgb_best_" + refit_target  + ".npz", confusion_mat)
    micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)

    # Change back to 1 maybe?
    exp_results = {
            "Accuracy":accuracy,
            "Precision":report["macro avg"]["precision"],
            "Recall":report["macro avg"]["recall"],
            "F1":report["macro avg"]["f1-score"],
            "Micro F1":micro_avg_f1,
            }
    print("Testing results were:")
    for k,v in exp_results.items():
        print(f"{k}:{v}")

    pd.DataFrame(hp_results.cv_results_).to_csv(base_dir+"/"+filename)

def validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, base_dir, filename="xgb_results.csv"):
    '''
    exp_num = 0
    y_test = np.array(y_test, dtype="int64")
    for max_depth in [3, 4]:
        for learning_rate in [0.1, 0.01, 0.001]:
            for gamma in list(range(0, 25, 5)) + [100, 1000]:
                for n_estimators in range(50, 400, 50):
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
                    break # FIXME
                break # FIXME
            break # FIXME

    print_best_results(base_dir+"/"+filename, "XGBoost", prob_type)
    '''
    y_test = np.array(y_test, dtype="int64")
    params = {
        'max_depth':[3, 4],
        'n_estimators':[50, 200, 400],
        'learning_rate':[0.1, 0.01, 0.001],
        'gamma': [0,10,25,100, 1000],
    }

    refit_target = 'neg_mean_squared_error'
    if OPT_STYLE == 'Bayes':
        print("Performing Bayesian Search")
        hp_tuner = BayesSearchCV(
               xgb.XGBRegressor(seed=573),
               search_spaces=params,
               cv=5,
               scoring=refit_target,
               refit=refit_target,
               )
    else:
        print("Performing Grid Search")
        hp_tuner = GridSearchCV(
               xgb.XGBRegressor(seed=573),
               param_grid=params,
               cv=5,
               scoring=['neg_mean_squared_error','neg_mean_absolute_error'],
               refit=refit_target,
               )
    hp_results = hp_tuner.fit(x_train, y_train)

    print(f"Best config found was {hp_results.best_params_} with the best {refit_target} of {hp_results.best_score_}.\n")

    xgb_reg = hp_results.best_estimator_
    preds = xgb_reg.predict(x_test)
    dump(xgb_reg, base_dir+"/xgb_reg/xgb_best_"+refit_target+".joblib")

    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    mae = sklearn.metrics.mean_absolute_error(y_test, preds)
    exp_results = {"MSE":mse, "MAE":mae}
    print("Testing results were:")
    for k,v in exp_results.items():
        print(f"{k}:{v}")
    pd.DataFrame(hp_results.cv_results_).to_csv(base_dir+"/"+filename)
