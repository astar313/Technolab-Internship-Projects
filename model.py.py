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

# In[1]:


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


# In[2]:


df = pd.read_csv('transfusion.data')


# In[3]:


df.head(10)


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.rename(columns={'whether he/she donated blood in March 2007':'Target'}, inplace=True)


# In[8]:


X = df.drop(columns=['Target'])


# In[9]:


X.shape


# In[10]:


y = df['Target']
y.head()

y.value_counts(normalize=True).round(3)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


# In[14]:


X_train.shape


# In[15]:


X_train.info()


# In[29]:


tpot = TPOTClassifier(
    generations=10,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)


# In[30]:


tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')


# In[31]:


print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


# In[32]:


tpot.fitted_pipeline_


# In[33]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=25.0, random_state=42)
#Fitting the model
logreg.fit(X_train,y_train)


# In[34]:


#Predicting on the test data
pred=logreg.predict(X_test)


# In[35]:


confusion_matrix(pred,y_test)


# In[36]:


logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


# In[37]:


#import pickle
#pickle.dump(logreg, open('model.pkl','wb'))


# In[38]:


#model=pickle.load(open('model.pkl','rb'))


# In[ ]:




