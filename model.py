#Importing require libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
import warnings
warnings.simplefilter("ignore")
df = pd.read_csv('transfusion.data')
df.head()
X = df.drop(columns=['whether he/she donated blood in March 2007'])
y = df['whether he/she donated blood in March 2007']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.229,stratify=y, random_state=42)

X_train.info()

tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')

tpot.fitted_pipeline_

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=25.0, random_state=42)
#Fitting the model
logreg.fit(X_train,y_train)

#Predicting on the test data
pred=logreg.predict(X_test)

confusion_matrix(pred,y_test)

logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')

import pickle
pickle.dump(logreg, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
