
# coding: utf-8

# In[2]:


import numpy as np
# import scipy as sp
# from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# import statsmodels.formula.api as smf
import pandas as pd
# import pyodbc
# import glob
import datetime
import glob
import os
import time

import seaborn as sns
sns.set_style("whitegrid")

from scipy.stats import wilcoxon, mannwhitneyu
from scipy import stats
import collections
import statsmodels.api as sm


# In[3]:


test_week_food1 = pd.read_pickle('/local/aling/test_week_food1.pkl')
control_week_food1= pd.read_pickle('/local/aling/control_week_food1.pkl')
test_week_food2 = pd.read_pickle('/local/aling/test_week_food2.pkl')
control_week_food2= pd.read_pickle('/local/aling/control_week_food2.pkl')
test_week_food3 = pd.read_pickle('/local/aling/test_week_food3.pkl')
control_week_food3= pd.read_pickle('/local/aling/control_week_food3.pkl')


# In[23]:


control_food123 = pd.concat([control_week_food1, control_week_food2, control_week_food3], axis=0, sort =True)


# In[25]:


test_food123 = pd.concat([test_week_food1, test_week_food2, test_week_food3], axis=0, sort =True)


# In[26]:


print(len(control_food123))
print(len(test_food123))


# In[27]:


control_food123 = control_food123.rename(index=str, columns={"entry_date": "dt"})
test_food123 = test_food123.rename(index=str, columns={"entry_date": "dt"})


# In[31]:


test_food123


# In[5]:


path = "/local/aling"
testweights2 = pd.read_pickle(path +'/testweightsbiweek.pkl')
controlweights2 = pd.read_pickle(path +'/controlweightsbiweek.pkl')


# In[6]:


controlid = set(controlweights2.common_user_id)
testid = set(testweights2.common_user_id)


# In[8]:


test_week_food1.dtypes


# In[28]:


control_weight_food123 = pd.merge(control_food123, controlweights2, on = ['common_user_id','dt'] )


# In[29]:


control_weight_food123.head()


# In[19]:


control_week_food1[control_week_food1.common_user_id == "00000b3d-d2f2-4ea6-bb4e-e9d770433e25"]


# In[20]:


controlweights2[controlweights2.common_user_id == "00000b3d-d2f2-4ea6-bb4e-e9d770433e25"]

