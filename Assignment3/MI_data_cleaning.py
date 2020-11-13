#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[12]:


data = pd.read_csv(r'D:\Projects\Machine_Intelligence\Assignment3\LBW_Dataset.csv')


# In[13]:


data.head()


# ### Replacing NaN values

# Age,Weight,HB,BP with mean

# In[14]:


replace_mean = ['Age','Weight','HB','BP']
for col in replace_mean:
    data[col].fillna(data[col].mean(),inplace = True)


# Delivery Phase,IFA,Residence with forward fill 

# In[15]:


forward_fill = ['Delivery phase','IFA' ,'Residence']
for col in forward_fill:
    data[col].ffill(axis = 0)


# Community,Education with mode

# In[16]:


#Education is being replaced with Nan, should check
replace_mode = ['Community','Education']
for col in replace_mode:
    data[col].fillna(data.mode()[col][0])


# ### Normalising the data 

# In[9]:


for column in data.columns:

    # Setting max 
    max_marks = max(data[column])

    # scaling the values to a range of 0 - 1
    scaled_values = []# list to store scaled values
 
    for item in range(len(data[column])):
        scaled_values.append(data[column][item] / max_marks)

    # updating the column with scaled_values
    data[column] = np.where(1,scaled_values, data[column])


# In[ ]:




