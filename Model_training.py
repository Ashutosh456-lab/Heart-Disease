import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import  RandomizedSearchCV, train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#################################################################################################################
##################################    Model Training     ######################################################
#################################################################################################################


with open(r'D:\Assignment\Heathcare PY files\X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open(r'D:\Assignment\Heathcare PY files\X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open(r'D:\Assignment\Heathcare PY files\y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open(r'D:\Assignment\Heathcare PY files\y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)



class ModelSelector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def random_forest(self):
        rf = RandomForestClassifier()
        params = {'n_estimators': [100, 300, 500], 'max_depth': [None, 5, 10]}
        clf = RandomizedSearchCV(rf, params, cv=5)
        clf.fit(self.X_train, self.y_train)
        return clf.best_estimator_
    
    def svm(self):
        svm = SVC()
        params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
        clf = RandomizedSearchCV(svm, params, cv=5)
        clf.fit(self.X_train, self.y_train)
        return clf.best_estimator_
    
    def logistic_regression(self):
        lr = LogisticRegression()
        params = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        clf = RandomizedSearchCV(lr, params, cv=5)
        clf.fit(self.X_train, self.y_train)
        return clf.best_estimator_
    
    def k_nearest_neighbors(self):
        knn = KNeighborsClassifier()
        params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        clf = RandomizedSearchCV(knn, params, cv=5)
        clf.fit(self.X_train, self.y_train)
        return clf.best_estimator_
    
    def decision_tree(self):
        dt = DecisionTreeClassifier()
        params = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}
        clf = RandomizedSearchCV(dt, params, cv=5)
        clf.fit(self.X_train, self.y_train)
        return clf.best_estimator_
    
    def compare_models(self):
        models = [self.random_forest(), self.svm(), self.logistic_regression(), self.k_nearest_neighbors(), self.decision_tree()]
        best_model = None
        best_accuracy = 0
        for model in models:
            y_pred = model.predict(self.X_train)
            accuracy = accuracy_score(self.y_train, y_pred)
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
        return best_model


# mc = ModelSelector(X_train, y_train)
# best_model = mc.logistic_regression()
# print(best_model)
# best_m=mc.compare_models()
# print(best_m)