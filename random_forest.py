import numpy as np
import pandas as pd
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from hyperparams import EXP_NAME, P_N_ESTIM, P_MAX_D, P_MAX_FEAT, P_N_ITER

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import scipy.stats.distributions as dists

PROB_NAME = {v:k for k,v in {'bank': 'binary_classification', 'maternal':'multiclass_classification', 'winequality':'regression'}.items()}

def validate_rf(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    if prob_type == "regression":
        validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir)
    else:
        validate_rf_classifier(x_train,  y_train, x_test, y_test, prob_type, tuner, notune, base_dir)

def validate_rf_classifier(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    exp_num = 0
    if notune:
        params = {'criterion':'gini','max_features':P_MAX_FEAT,'max_depth':P_MAX_D,'n_estimators':P_N_ESTIM,'class_weight':'balanced','random_state':573,'n_jobs':-1}
        rf_clf = RandomForestClassifier(**params)
        rf_clf.fit(x_train, y_train)
        # In case we do inference experiments
        #dump(rf_clf, base_dir+"/rf_clf/rf_" + str(EXP_NAME) + ".joblib")
        preds = rf_clf.predict(x_test)

        report = classification_report(y_test, preds, output_dict=True)
        confusion_mat = confusion_matrix(y_test, preds)
        np.savez(base_dir + "/rf_results/rf_" + str(EXP_NAME) + ".npz", confusion_mat)
        micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)

        exp_results = [accuracy,report["macro avg"]["precision"],report["macro avg"]["recall"],report["macro avg"]["f1-score"],micro_avg_f1]
        exp_results = [str(x) for x in exp_results]
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_MAX_FEAT)]+exp_results), file=open(f'percomb_runs/results_rf_{PROB_NAME[prob_type]}.csv','a'))
    else:
        # Maybe change this to weighted or samples based f1
        scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_macro','recall_macro','f1_macro', 'f1_micro']
        refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

        hp_tuner = None
        if tuner == 'bayes':
            params = {
                'max_depth':Integer(3,5),
                'n_estimators':Integer(50,150),
                'max_features':Real(0.2,0.8),
            }
            hp_tuner = BayesSearchCV(
                RandomForestClassifier(criterion='gini',class_weight='balanced',random_state=573,n_jobs=-1),
                search_spaces=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                n_iter=P_N_ITER,
                random_state=573,
                )
        if tuner == 'random':
            params = {
                'max_depth':dists.randint(3,5+1),
                'n_estimators':dists.randint(50,150+1),
                'max_features':dists.uniform(0.2,0.8),
            }
            hp_tuner = RandomizedSearchCV(
                RandomForestClassifier(criterion='gini',class_weight='balanced',random_state=573,n_jobs=-1),
                param_distributions=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                n_iter=P_N_ITER,
                random_state=573,
                )
        elif tuner == 'grid':
            params = {
                'max_depth':[3, 4, 5],
                'n_estimators':[50, 100, 150],
                'max_features':[0.2,0.5,0.8],
            }

            hp_tuner = GridSearchCV(
                RandomForestClassifier(criterion='gini',class_weight='balanced',random_state=573,n_jobs=-1),
                param_grid=params,
                cv=5,
                scoring=scoring_options,
                refit=refit_target,
                )
        hp_results = hp_tuner.fit(x_train, y_train)

        rf_clf = hp_results.best_estimator_
        preds = rf_clf.predict(x_test)

        #dump(rf_clf, base_dir+"/rf_clf/rf_best_" + refit_target + ".joblib")
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
        to_print = [str(x) for x in [tuner, str(P_N_ITER)]+list(hp_results.best_params_.values())+list(exp_results.values())+[refit_target]]
        print(",".join(to_print), file=open(f"bayes_runs/results_rf_{PROB_NAME[prob_type]}.csv",'a'))

        pd.DataFrame(hp_results.cv_results_).to_csv(f"bayes_runs/all_combs/rf_{P_N_ITER}_{PROB_NAME[prob_type]}_{tuner}.csv")

def validate_rf_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="rf_results.csv"):
    if notune:
        params = {"criterion":"squared_error","max_features":P_MAX_FEAT, 'max_depth':P_MAX_D, 'n_estimators':P_N_ESTIM, 'random_state':573, 'n_jobs':-1}
        rf_reg = RandomForestRegressor(**params)
        rf_reg.fit(x_train, y_train)
        # In case we do inference tests
        #dump(rf_reg, base_dir+"/rf_reg/rf_" + str(EXP_NAME) + ".joblib")
        preds = rf_reg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)

        exp_results = [str(mse),str(mae)]
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_MAX_FEAT)]+exp_results), file=open(f'percomb_runs/results_rf_{PROB_NAME[prob_type]}.csv','a'))
    else:
        refit_target = 'neg_mean_squared_error'
        hp_tuner = None
        if tuner == 'bayes':
            params = {
                'max_depth':Integer(3,5),
                'n_estimators':Integer(50,150),
                'max_features':Real(0.2,0.8),
            }
            hp_tuner = BayesSearchCV(
                RandomForestRegressor(criterion="squared_error",random_state=573,n_jobs=-1),
                search_spaces=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                random_state=573,
                n_iter=P_N_ITER,
                )
        if tuner == 'random':
            params = {
                'max_depth':dists.randint(3,5+1),
                'n_estimators':dists.randint(50,150+1),
                'max_features':dists.uniform(0.2,0.8),
            }
            hp_tuner = RandomizedSearchCV(
                RandomForestRegressor(criterion="squared_error",random_state=573,n_jobs=-1),
                param_distributions=params,
                cv=5,
                scoring=refit_target,
                refit=refit_target,
                random_state=573,
                n_iter=P_N_ITER,
                )
        elif tuner == 'grid':
            params = {
                #'criterion':["squared_error", "absolute_error", "poisson"],
                'max_depth':[3, 4, 5],
                'n_estimators':[50, 100, 150],
                'max_features':[0.2,0.5,0.8],
            }
            hp_tuner = GridSearchCV(
                RandomForestRegressor(criterion="squared_error",random_state=573,n_jobs=-1),
                param_grid=params,
                cv=5,
                scoring=['neg_mean_squared_error','neg_mean_absolute_error'],
                refit=refit_target,
                )
        hp_results = hp_tuner.fit(x_train, y_train)
        rf_reg = hp_results.best_estimator_
        preds = rf_reg.predict(x_test)

        #dump(rf_reg, base_dir+"/rf_reg/rf_best_"+refit_target+".joblib")
        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)

        exp_results = {"MSE":mse, "MAE":mae}
        to_print = [str(x) for x in [tuner,str(P_N_ITER)]+list(hp_results.best_params_.values())+list(exp_results.values())+["MSE"]]
        print(",".join(to_print), file=open(f"bayes_runs/results_rf_{PROB_NAME[prob_type]}.csv",'a'))

        pd.DataFrame(hp_results.cv_results_).to_csv(f"bayes_runs/all_combs/rf_{P_N_ITER}_{PROB_NAME[prob_type]}_{tuner}.csv")
