
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

#################################################################################################################
##################################    Feature Engineering     ####################################################
#################################################################################################################

# 1. Outlier Handling
# 2. Scaling

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=3.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in X.columns:
            if X[i].dtype != 'O':
                median = X[i].median()
                std = X[i].std()
                outlier_min = median - self.thresh * std
                outlier_max = median + self.thresh * std
                X[i][(X[i] < outlier_min) | (X[i] > outlier_max)] = median
        return X


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_thresh=3.0):
        self.outlier_thresh = outlier_thresh
        self.outlier_replacer = OutlierReplacer(thresh=self.outlier_thresh)
        self.scaler = StandardScaler()
        self.preprocessor = ColumnTransformer(transformers=[
            ('outlier_replacer', self.outlier_replacer, ['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                                         'thalach', 'exang','oldpeak', 'slope','ca', 'thal']),
            ('scaler', self.scaler, ['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                     'oldpeak', 'slope','ca', 'thal']),
        ])

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)



























