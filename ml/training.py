from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import lightgbm as lgb

def train_baseline(X_train, y_train, X_test, y_test):
    y_pred = y_train['y'].mean()
    y_test = np.array(y_test['y'])
    y_test = y_test[~np.isnan(y_test)]
    mse = mean_squared_error(y_test, [y_pred]*len(y_test))
    return math.sqrt(mse)


def score_cv(model, X_train, y_train, X_test, y_test):
    fitted_model = model.fit(X_train, y_train)
    y_pred = fitted_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return math.sqrt(mse)


def define_models():
    print(" > Defining models to train...")
    models = dict()

    ## Linear
    linear = make_pipeline(RobustScaler(), LinearRegression())
    models['linear'] = linear

    ## Lasso
    alpha = [0.7, 1, 16]
    for a in alpha:
        lasso = make_pipeline(RobustScaler(), Lasso(random_state=1, alpha=a, max_iter=10000, tol=0.001))
        models['lasso, alpha={a}'.format(a=a)] = lasso

    ## Elastic Net
    # ENet = make_pipeline(RobustScaler(), ElasticNet(random_state=1))
    # models['Elastic Net'] = ENet

    ## Bayesian Ridge
    bayesian = make_pipeline(RobustScaler(), BayesianRidge())
    models['Bayesian Ridge'] = bayesian

    ## Random Forest
    # max_depth = [1, 16, 64, 256]
    # min_samples_split = [2, 64]
    # min_samples_leaf = [2, 64]
    # for d in max_depth:
    #     for s in min_samples_split:
    #         for l in min_samples_leaf:
    #             rf = RandomForestRegressor(random_state=1, max_depth=d, min_samples_split=s, n_estimators=500, n_jobs=4)
    #             models['Random Forest, max_depth={d}, min_samples_split={s}, min_samples_leaf={l}'.format(d=d, s=s, l=l)] = rf

    ## Gradient Boost
    learning_rate = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    for l in learning_rate:
        GBoost = GradientBoostingRegressor(random_state=1, learning_rate=l)
        models['Gradient Boosting, lr={l}'.format(l=l)] = GBoost

    # ## XG Boost
    # eta = [1, 0, 0.3, 0.66, 1]
    # gamma = [0, 0.1, 4, 16, 64]
    # max_depth = [1, 4, 16, 64, 256]
    # min_child_weight = [0, 0.1, 4, 16, 64]
    # for e in eta:
    #     for g in gamma:
    #         for d in max_depth:
    #             for c in min_child_weight:
    #                 model_xgb = xgb.XGBRegressor(random_state=1, eta=e, gamma=g, max_depth=d, min_child_weight=c)
    #                 models['XG Boost, eta={e}, gamma={g}, max_depth={d}, min_child_weight={c}'.format(e=e,g=g,d=d,c=c)] = model_xgb


    return models


def train_pipeline(X_train, y_train, X_test, y_test, models):
    # prepare models
    y_train = np.array(y_train['y'])
    y_test = np.array(y_test['y'])
    X_train = np.array(X_train) 
    X_test = np.array(X_test) 

    # train and score models
    scores = []
    for m in models.keys():
        scores.append(score_cv(models[m], X_train, y_train, X_test, y_test))
    return scores, models


def train(datalist):
    X_trains, y_trains, X_tests, y_tests, groups = datalist

    models = define_models()
    n_models = len(models.keys())
    rmse = np.zeros((len(X_trains),n_models+1))
    i = 0
    print(" > Start training...")
    for i in range(len(X_trains)):
        if (len(X_trains[i])!=0) & (len(X_tests[i])!=0):
            rmse[i,0] = train_baseline(X_trains[i], y_trains[i], X_tests[i], y_tests[i])
            scores, models = train_pipeline(X_trains[i], y_trains[i], X_tests[i], y_tests[i], models)
            rmse[i,1:len(scores)+1] = scores
            i+=1
            print(" > Finished time split {i}/5.".format(i=i))
    print(" > Training finished.")
    return rmse, ['baseline'] + list(models.keys()), models