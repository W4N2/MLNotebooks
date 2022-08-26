"""Utility functions for ML Models"""
import pandas as pd
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class MLUtils(object):
    def main(self, X: any, y: any):
        """ Init class for ml utils

        Args:
            X (any): x parameters
            y (any): target variable
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.3, random_state=1)

        
        x_train_num_cols = [col for col in self.X_train.columns if self.X_train[col].dtypes in ("int64", "float64")]
        x_train_cat_cols = [col for col in self.X_train.columns if self.X_train[col].dtypes in ("object", "category")]
        ohe = OneHotEncoder()
        mms = MinMaxScaler()

        t = [
            ('cat', ohe, x_train_cat_cols), 
            ('num', mms, x_train_num_cols)
            ]
        self.col_transform = ColumnTransformer(transformers=t)
        self.polfeatures = PolynomialFeatures()

        return self.grid_search(self.X_train, self.y_train, self.X_test, self.y_test, self.col_transform, self.polfeatures)
        

    def grid_search(self, X_train, y_train, X_test, y_test, col_transform, polfeatures):

        model_params = {
            'GaussianNB': {
                'model': GaussianNB(priors= None, var_smoothing = 1e-09), 
                'params': {
                    } 
            },
            'random_forest' : {
                'model': RandomForestClassifier(),
                'params' : {
                    'n_estimators': [1,5,10],
                }
            },
            'logistic_regression' : {
                'model': LogisticRegression(solver='liblinear',multi_class='auto'),
                'params': {
                    'C': [1,5, 10, 15],
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params' :{
                    'n_neighbors': [3, 5, 6, 7, 8, 9, 10]
                }
            },
            'XGBoost' : {
                'model': XGBClassifier(random_state = 1),
                'params': {
                    'max_depth': [1,3,5]
                }
            },
            'SVC': {
                'model': SVC(),
                'params': {
                    'kernel': ['linear']
                }
            }
        }

        scores = []
        for model_name, mp in model_params.items():
            model = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
            pipeline = Pipeline(steps=[
            ('t', col_transform), 
            ("p", polfeatures),
            ('m', model)
            ], )
            pipeline.fit(X_train, y_train)
            y_predicted = pipeline.predict(X_test)
            scores.append({
                'model': model_name,
                'best_score': model.best_score_,
                'best_params': model.best_params_,
                'best_recall': metrics.recall_score(y_test, y_predicted)
            })
        
        self.dff = pd.DataFrame(scores,columns=['model','best_score','best_params', 'best_recall'])
        return self.dff
