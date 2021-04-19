from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import numpy as np 
from sklearn.model_selection import train_test_split

class SimpleModel():
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.features = [c for c in X]
        self.model.fit(X, y)

    def predict(self, X):
        X_test = X[self.features]
        y_pred = self.model.predict(X_test)
        return y_pred
    
class BoostingModel():
    def __init__(self):
        self.model = lgb
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
            'verbose': -1
        }

    def fit(self, X, y):
        # define weights  
        w = [1 if y.iloc[i]==0 else 1e8 for i in range(len(y.values))]
        # define training data (whole dataset)
        lgb_train = self.model.Dataset(X, y, free_raw_data=False, weight=w)
        # train the model
        self.model = self.model.train(
                self.params,
                lgb_train,
                num_boost_round=1000
               )
        
    def predict(self, X):
        # make prediction
        y_pred = np.array(self.model.predict(X) > 0.9, dtype=int)
        return y_pred
