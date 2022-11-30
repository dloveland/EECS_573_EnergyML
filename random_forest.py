"""
FIXME
"""

import numpy as np
import pandas as pd
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from hyperparams import EXP_NAME, P_N_ESTIM, P_MAX_D, P_MAX_FEAT, P_LR

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

def validate_rf(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    if prob_type == "regression":
        validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir)
    else:
        validate_rf_classifier(x_train,  y_train, x_test, y_test, prob_type, tuner, notune, base_dir)

def validate_rf_classifier(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    exp_num = 0
    if notune:
        #print("Performing training using appropriate params from hyperparam.txt")
        params = {'criterion':'gini','max_features':P_MAX_FEAT, 'max_depth':P_MAX_D, 'n_estimators':P_N_ESTIM, 'class_weight':'balanced','random_state':573,'n_jobs':-1}
        #print(f'Params are {params}')
        #input("Confirm and press enter")
        rf_clf = RandomForestClassifier(**params)
        rf_clf.fit(x_train, y_train)
        dump(rf_clf, base_dir+"/rf_clf/rf_" + str(EXP_NAME) + ".joblib")
        preds = rf_clf.predict(x_test)
        report = classification_report(y_test, preds, output_dict=True)
        confusion_mat = confusion_matrix(y_test, preds)
        np.savez(base_dir + "/rf_results/rf_" + str(EXP_NAME) + ".npz", confusion_mat)
        micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)
        exp_results = [accuracy,report["macro avg"]["precision"],report["macro avg"]["recall"],report["macro avg"]["f1-score"],micro_avg_f1]
        exp_results = [str(x) for x in exp_results]
        #print("Done with experiment " + str(EXP_NAME))
        #for k,v in exp_results.items():
        #    print(f"{k}:{v}")
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_MAX_FEAT)]+exp_results))
    else:
        params = {
            #'criterion':["gini", "entropy", "log_loss"],
            'max_depth':[3, 4, 5],
            'n_estimators':[50, 100, 150],
            'max_features':[0.25,0.5,1],
        }

        # Maybe change this to weighted or samples based f1
        scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_macro','recall_macro','f1_macro', 'f1_micro']
        refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

        hp_tuner = None
        if tuner == 'bayes':
            print("Performing Bayesian Search")
            hp_tuner = BayesSearchCV(
                RandomForestClassifier(criterion='gini',class_weight='balanced',random_state=573,n_jobs=-1),
                search_spaces=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                random_state=573,
                )
        else:
            print("Performing Grid Search")
            hp_tuner = GridSearchCV(
                RandomForestClassifier(criterion='gini',class_weight='balanced',random_state=573,n_jobs=-1),
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
        print("Testing results were:")
        for k,v in exp_results.items():
            print(f"{k}:{v}")

        pd.DataFrame(hp_results.cv_results_).to_csv(base_dir+"/"+filename)

def validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    if notune:
        #print("Performing training using appropriate params from hyperparam.txt")
        params = {"criterion":"squared_error","max_features":P_MAX_FEAT, 'max_depth':P_MAX_D, 'n_estimators':P_N_ESTIM, 'random_state':573, 'n_jobs':-1}
        #print(f'Params are {params}')
        rf_reg = RandomForestRegressor(**params)
        rf_reg.fit(x_train, y_train)
        dump(rf_reg, base_dir+"/rf_reg/rf_" + str(EXP_NAME) + ".joblib")
        preds = rf_reg.predict(x_test)
        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)
        exp_results = [str(mse),str(mae)]
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_LR)]+exp_results))
    else:
        params = {
            #'criterion':["squared_error", "absolute_error", "poisson"],
            'max_depth':[3, 4, 5],
            'n_estimators':[50, 100, 150],
            'max_features':[0.25,0.5,1],
        }
        refit_target = 'neg_mean_squared_error'
        hp_tuner = None
        if tuner == 'bayes':
            print("Performing Bayesian Search")
            hp_tuner = BayesSearchCV(
                RandomForestRegressor(criterion="squared_error",random_state=573,n_jobs=-1),
                search_spaces=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                random_state=573,
                )
        else:
            print("Performing Grid Search")
            hp_tuner = GridSearchCV(
                RandomForestClassifier(criterion="squared_error",random_state=573,n_jobs=-1),
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
