#!/usr/bin/env python
# coding: utf-8

# Project Tasks
# 
# 1. Inspecting transfusion.data file
# 2. Loading the blood donations data
# 3. Inspecting transfusion DataFrame
# 4. Creating target column
# 5. Checking target incidence
# 6. Splitting transfusion into train and test datasets
# 7. Selecting model using TPOT
# 8. Checking the variance
# 9. Log normalization
# 10. Training the linear regression model
# 11. Conclusion

# In[250]:


#Importing require libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
import warnings
warnings.simplefilter("ignore")


# In[251]:


df = pd.read_csv('transfusion.data')


# In[252]:


df.head()


# In[253]:


df.describe()


# In[254]:


df.shape


# In[255]:


X = df.drop(columns=['whether he/she donated blood in March 2007'])


# In[256]:


X.shape


# In[257]:


y = df['whether he/she donated blood in March 2007']
y.head()


# In[265]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.229,stratify=y, random_state=42)


# In[266]:


X_train.shape


# In[267]:


X_train.info()


# In[286]:


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


# In[287]:


tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')


# In[288]:


print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


# In[223]:


tpot.fitted_pipeline_


# In[292]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=25.0, random_state=42)
#Fitting the model
logreg.fit(X_train,y_train)


# In[294]:


#Predicting on the test data
pred=logreg.predict(X_test)


# In[296]:


confusion_matrix(pred,y_test)


# In[299]:


logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[300]:


import pickle
pickle.dump(logreg, open('model.pkl','wb'))


# In[301]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:




