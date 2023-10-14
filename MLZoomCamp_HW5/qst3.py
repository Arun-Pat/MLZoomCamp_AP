#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


# In[3]:


dv = load('dv.bin')
model = load('model1.bin')

# client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




