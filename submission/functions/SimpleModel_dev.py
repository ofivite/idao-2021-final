from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

class BoostingModel_dev():
    def __init__(self, loss=None):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 4,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 777,
            'metric': 'auc',
#             'early_stopping': 10,
            'verbose': -1
        }
        self.fobj, self.feval = None, None
        if loss:
            self.fobj, self.feval = loss.lgb_obj, loss.lgb_eval

    def fit(self, X_train, y_train, num_boost_round=1000, w0=None, w1=None, X_val=None, y_val=None, early_stopping=None):
        self.params['early_stopping'] = early_stopping
        w_train, w_train = None, None
        if w0 is not None and w1 is not None:
            w_train = np.where(y_train == 0, w0, w1)
        lgb_train = lgb.Dataset(X_train, y_train, weight=w_train, free_raw_data=False)
        lgb_val = None
        if X_val is not None and y_val is not None:
            if w0 is not None and w1 is not None:
                w_val = np.where(y_val == 0, w0, w1)
            lgb_val = lgb.Dataset(X_val, y_val, weight=w_val, reference=lgb_train, free_raw_data=False)
        self.model = lgb.train(self.params, lgb_train, num_boost_round=num_boost_round, valid_sets=lgb_val,
                              fobj=self.fobj, feval=self.feval
                              )

    def predict(self, X):
#         y_pred = np.array(self.model.predict(X) > 1 - 1e-1, dtype=int)
        y_pred = self.model.predict(X)
        return y_pred
