#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install joblib


# In[2]:


import pandas as pd
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[4]:


df = pd.read_csv("diabetes.csv")
df


# In[7]:


x = df.drop('class', axis=1)
y = df['class']


# In[8]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled


# In[9]:


model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)


# In[10]:


print("Cross-validation scores:", scores)
print("Mean accuracy:",scores.mean())


# In[11]:


model.fit(X_scaled, y)

joblib.dump(model, 'model.pk1')
joblib.dump(scaler, 'scaler.pk1')


# In[ ]:




