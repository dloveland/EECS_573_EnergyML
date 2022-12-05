import numpy as np
import sklearn
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from hyperparams import EXP_NAME, P_N_ESTIM, P_MAX_D, P_LR, P_COL_BT, P_N_ITER
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from file_utils import *
import scipy.stats.distributions as dists

PROB_NAME = {v:k for k,v in {'bank': 'binary_classification', 'maternal':'multiclass_classification', 'winequality':'regression'}.items()}

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
                 'colsample_bytree':P_COL_BT,
                 'eval_metric':'logloss',
                 'objective':obj,
                 'seed':573
        }
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(x_train, y_train)
        # If we do inference runs will need this. Remove for now to not pollute energy cost;
        #dump(xgb_clf, base_dir+"/xgb_clf/xgb_" + str(EXP_NAME) + ".joblib")
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
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_LR),str(P_COL_BT)]+exp_results), file=open(f'percomb_runs/results_xgb_{PROB_NAME[prob_type]}.csv','a'))
        return
    else:
        scoring_options = ['accuracy','precision','recall','f1','f1_micro'] if 'binary' in prob_type else ['accuracy','precision_macro','recall_macro','f1_macro', 'f1_micro']
        refit_target = 'f1' if 'binary' in prob_type else 'f1_macro'

        if tuner == 'bayes':
            params = {
                'max_depth':Integer(3,5),
                'n_estimators':Integer(50,150),
                'colsample_bytree':Real(0.2,0.8),
            }
            hp_tuner = BayesSearchCV(
                   xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
                   search_spaces=params,
                   cv=5,
                   n_iter=P_N_ITER,
                   scoring=refit_target,
                   refit=refit_target,
            )
        if tuner == 'random':
            params = {
                'max_depth':dists.randint(3,5+1),
                'n_estimators':dists.randint(50,150+1),
                'colsample_bytree':dists.uniform(0.2,0.8),
            }
            hp_tuner = RandomizedSearchCV(
                   xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
                   param_distributions=params,
                   cv=5,
                   n_iter=P_N_ITER,
                   scoring=refit_target,
                   refit=refit_target,
            )
        elif tuner == 'grid':
            params = {
                'max_depth':[3, 4, 5],
                'n_estimators':[50, 100, 150],
                'colsample_bytree':[0.2, 0.5, 0.8],
            }
            hp_tuner = GridSearchCV(
                   xgb.XGBClassifier(objective=obj,eval_metric='logloss',seed=573),
                   param_grid=params,
                   cv=5,
                   scoring=scoring_options,
                   refit=refit_target,
            )
        hp_results = hp_tuner.fit(x_train, y_train)

        xgb_clf = hp_results.best_estimator_
        preds = xgb_clf.predict(x_test)

        #dump(xgb_clf, base_dir+"/xgb_results/xgb_best_" + refit_target + ".joblib")
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
        to_print = [str(x) for x in [tuner,str(P_N_ITER)]+list(hp_results.best_params_.values())+list(exp_results.values())+[refit_target]]
        print(",".join(to_print), file=open(f"bayes_runs/results_xgb_{PROB_NAME[prob_type]}.csv","a"))

        pd.DataFrame(hp_results.cv_results_).to_csv(f"bayes_runs/all_combs/xgb_{P_N_ITER}_{PROB_NAME[prob_type]}_{tuner}.csv")

def validate_xgb_regressor(x_train, y_train, x_test, y_test, prob_type, tuner, notune,base_dir, filename="xgb_results.csv"):
    y_test = np.array(y_test, dtype="int64")
    if notune:
        params = {'max_depth':P_MAX_D,
                 'n_estimators':P_N_ESTIM,
                 'learning_rate':P_LR,
                 'colsample_bytree':P_COL_BT,
                 'seed':573,
        }
        xgb_clf = xgb.XGBRegressor(**params)
        xgb_clf.fit(x_train, y_train)
        # If we do inference runs
        #dump(xgb_clf, base_dir+"/xgb_reg/xgb_" + str(EXP_NAME) + ".joblib")
        preds = xgb_clf.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)

        exp_results = [str(mse),str(mae)]
        print(','.join([str(EXP_NAME),str(P_MAX_D),str(P_N_ESTIM),str(P_LR),str(P_COL_BT)]+exp_results), file=open(f'percomb_runs/results_xgb_{PROB_NAME[prob_type]}.csv','a'))
        return
    else:
        refit_target = 'neg_mean_squared_error'
        if tuner == 'bayes':
            params = {
                'max_depth':Integer(3,5),
                'n_estimators':Integer(50,150),
                'colsample_bytree':Real(0.2,0.8),
            }

            hp_tuner = BayesSearchCV(
                   xgb.XGBRegressor(seed=573),
                   search_spaces=params,
                   cv=5,
                   n_iter=P_N_ITER,
                   scoring=refit_target,
                   refit=refit_target,
                   )
        if tuner == 'random':
            params = {
                'max_depth':dists.randint(3,5+1),
                'n_estimators':dists.randint(50,150+1),
                'colsample_bytree':dists.uniform(0.2,0.8),
            }

            hp_tuner = RandomizedSearchCV(
                   xgb.XGBRegressor(seed=573),
                   param_distributions=params,
                   cv=5,
                   n_iter=P_N_ITER,
                   scoring=refit_target,
                   refit=refit_target,
                   )
        elif tuner == "grid":
            params = {
                'max_depth':[3, 4, 5],
                'n_estimators':[50, 100, 150],
                'colsample_bytree':[0.2,0.5,0.8],
            }
            hp_tuner = GridSearchCV(
                   xgb.XGBRegressor(seed=573),
                   param_grid=params,
                   cv=5,
                   scoring=['neg_mean_squared_error','neg_mean_absolute_error'],
                   refit=refit_target,
                   )
        hp_results = hp_tuner.fit(x_train, y_train)

        xgb_reg = hp_results.best_estimator_
        preds = xgb_reg.predict(x_test)

        mse = sklearn.metrics.mean_squared_error(y_test, preds)
        mae = sklearn.metrics.mean_absolute_error(y_test, preds)

        exp_results = {"MSE":mse, "MAE":mae}
        to_print = [str(x) for x in [tuner,str(P_N_ITER)]+list(hp_results.best_params_.values())+list(exp_results.values())+["MSE"]]
        print(",".join(to_print), file=open(f"bayes_runs/results_xgb_{PROB_NAME[prob_type]}.csv","a"))

        pd.DataFrame(hp_results.cv_results_).to_csv(f"bayes_runs/all_combs/xgb_{P_N_ITER}_{PROB_NAME[prob_type]}_{tuner}.csv")
