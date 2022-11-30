import numpy as np
import sklearn
from file_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from hyperparams import EXP_NAME, P_N_ESTIM, P_MAX_D, P_LR

import xgboost as xgb
from skopt import BayesSearchCV

def validate_xgb(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="xgb_results.csv"):
    if prob_type == "regression":
        validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir)
    else:
        validate_xgb_classifier(x_train,  y_train, x_test, y_test, prob_type, tuner, notune, base_dir)

def validate_xgb_classifier(x_train, y_train, x_test, y_test, prob_type, tuner, notune, base_dir, filename="xgb_results.csv"):
    y_test = np.array(y_test, dtype="int64")
    obj = 'binary:logistic' if 'binary' in prob_type else 'multi_softmax'
    if notune:
        params = {'max_depth':P_MAX_D,
                 'n_estimators':P_N_ESTIM,
                 'learning_rate':P_LR,
                 'eval_metric':'logloss',
                 'objective':obj,
                 'seed':573
        }
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(x_train, y_train)
        dump(xgb_clf, base_dir+"/xgb_clf/xgb_" + str(EXP_NAME) + ".joblib")
        preds = xgb_clf.predict(x_test)
        report = classification_report(y_test, preds, output_dict=True)
        confusion_mat = confusion_matrix(y_test, preds)
        np.savez(base_dir + "/xgb_results/xgb_" + str(EXP_NAME) + ".npz", confusion_mat)
        micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)

        exp_results = {"Accuracy":accuracy,
                "Precision":report["macro avg"]["precision"],
                "Recall":report["macro avg"]["recall"],
                "F1":report["macro avg"]["f1-score"],
                "Micro F1":micro_avg_f1
        }

        exp_results = [accuracy,report["macro avg"]["precision"],report["macro avg"]["recall"],report["macro avg"]["f1-score"],micro_avg_f1]
        exp_results = [str(x) for x in exp_results]
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_LR)]+exp_results))
        return
    else:
        params = {
            'max_depth':[3, 4, 5],
            'n_estimators':[50, 100, 150],
            'learning_rate':[0.1, 0.01, 0.001],
        }
        scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_macro','recall_macro','f1_macro', 'f1_micro']
        refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

        if tuner == 'bayes':
            print("Performing Bayesian Search")
            hp_tuner = BayesSearchCV(
                   xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
                   search_spaces=params,
                   cv=5,
                   scoring=refit_target,
                   refit=refit_target,
                   random_state=573,
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

        xgb_clf = hp_results.best_estimator_
        preds = xgb_clf.predict(x_test)
        dump(xgb_clf, base_dir+"/xgb_results/xgb_best_" + refit_target + ".joblib")
        report = classification_report(y_test, preds, output_dict=True)
        confusion_mat = confusion_matrix(y_test, preds)
        np.savez(base_dir + "/xgb_results/xgb_best_" + refit_target  + ".npz", confusion_mat)
        micro_avg_f1 = sklearn.metrics.f1_score(y_test, preds, average="micro")
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)

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

def validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune,base_dir, filename="xgb_results.csv"):
    y_test = np.array(y_test, dtype="int64")
    if notune:
        params = {'max_depth':P_MAX_D,
                 'n_estimators':P_N_ESTIM,
                 'learning_rate':P_LR,
                 'seed':573,
        }
        xgb_clf = xgb.XGBRegressor(**params)
        xgb_clf.fit(x_train, y_train)
        dump(xgb_clf, base_dir+"/xgb_reg/xgb_" + str(EXP_NAME) + ".joblib")
        preds = xgb_clf.predict(x_test)
        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)

        exp_results = [str(mse),str(mae)]
        #for k,v in exp_results.items():
        #    print(f"{k}:{v}")
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_LR)]+exp_results))
        return
    else:
        params = {
            'max_depth':[3, 4, 5],
            'n_estimators':[50, 100, 150],
            'learning_rate':[0.1, 0.01, 0.001],
        }

        refit_target = 'neg_mean_squared_error'
        if tuner == 'bayes':
            print("Performing Bayesian Search")
            hp_tuner = BayesSearchCV(
                   xgb.XGBRegressor(seed=573),
                   search_spaces=params,
                   cv=5,
                   scoring=refit_target,
                   refit=refit_target,
                   random_state=573, 
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
