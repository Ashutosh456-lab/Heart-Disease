import pickle
from sklearn.pipeline import Pipeline
import sys
sys.path.append(r'C:\Users\ashutosh_lande\Downloads\Heathcare PY files\Heathcare PY files')
import pandas as pd
import numpy as np
from Model_training import ModelSelector
from data_preprocessing import Preprocessor
#################################################################################################################
##################################          Pipeline         ####################################################
#################################################################################################################
with open(r'C:\Users\ashutosh_lande\Downloads\Heathcare PY files\Heathcare PY files\X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open(r'C:\Users\ashutosh_lande\Downloads\Heathcare PY files\Heathcare PY files\X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open(r'C:\Users\ashutosh_lande\Downloads\Heathcare PY files\Heathcare PY files\y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open(r'C:\Users\ashutosh_lande\Downloads\Heathcare PY files\Heathcare PY files\y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
    

preprocessor = Preprocessor(outlier_thresh=3.0)
mc = ModelSelector(X_train, y_train)
best_m=mc.compare_models()
print(best_m)

pipeline = Pipeline([('preprocessor', preprocessor),('clf',best_m )])

pipeline.fit(X_train, y_train)

######################################  Testing   Data Pass     ####################################################

new_data=np.array([[57,	0,	0	,140,	241,	0,	1	,123,	1,	0.2,	1,	0	,3		]])

new_data = pd.DataFrame(new_data, columns=['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                           'thalach', 'exang', 'oldpeak', 'slope','ca', 'thal'])


######################################        Predictions        ################################################
predictions = pipeline.predict(new_data)
print(predictions[0])
  
if predictions== 0:
    print("Result: No Heart Disease") 
    
elif predictions ==1:
    print('Result: Heart Disease')







