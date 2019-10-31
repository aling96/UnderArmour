
# coding: utf-8

# In[1]:


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
from datetime import datetime
import glob
import os
import time

import seaborn as sns
sns.set_style("whitegrid")

from scipy.stats import wilcoxon, mannwhitneyu
from scipy import stats
from collections import Counter
import statsmodels.api as sm


# In[ ]:


path = "/local/aling"
start_time = time.time()
weights2 = pd.read_pickle(path +'/weights2.pkl')
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


path = "/local/aling"
start_time = time.time()
low_daily_workouts = pd.read_pickle(path +'/low_daily_workouts.pkl')
print("--- %s seconds ---" % (time.time() - start_time))


# In[3]:


path = "/local/aling"
start_time = time.time()
workouts = pd.read_pickle(path +'/cutworkouts.pkl')
print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:


workouts.head()


# In[ ]:


#fullweights.head()


# In[ ]:


#print(len(fullweights))
#print(min(fullweights.dt))
#print(max(fullweights.dt))


# In[ ]:


print(len(low_daily_workouts))
print(min(low_daily_workouts.workout_date))
print(max(low_daily_workouts.workout_date))


# In[ ]:


#print(len(fullworkouts))
#print(min(fullworkouts.workout_date))
#print(max(fullworkouts.workout_date))


# In[ ]:


#fullworkouts['workout_date'] = pd.to_datetime(fullworkouts.workout_date, format = '%Y-%m-%d', errors = 'coerce')


# In[ ]:


#fullweights['dt'] = pd.to_datetime(fullweights.dt, format = '%Y-%m-%d', errors = 'coerce')
#fullweights = fullweights.rename(index = str, columns = {'user_id':'common_user_id'})


# In[ ]:


#sum(fullweights.dt >= '2014-01-01')


# In[ ]:


#workouts = fullworkouts[(fullworkouts.workout_date >= '2008-01-25') & (fullworkouts.workout_date <= '2017-04-21')]


# In[ ]:


#workouts.to_pickle('/local/aling/cutworkouts.pkl')


# In[ ]:


# twos = weights2['bmicat'] == 2 #find all 2s
# twos1 = weights2.common_user_id[twos].reset_index(drop=True) #id numbers
# normalbmi = weights2[weights2.common_user_id.isin(twos1)]


# In[ ]:


# normalbmi = normalbmi.groupby(['common_user_id', pd.Grouper(key='dt', freq='D')])[['bmi','bmicat']].first().reset_index().sort_values(['common_user_id','dt'])
# normalbmi['dt_diff'] = normalbmi.groupby('common_user_id')['dt'].apply(lambda x: x.diff()).dt.days
# # normalbmi['dt_diff'] = normalbmi['dt_diff'].fillna(0)


# In[ ]:


# #count number of times a user has been in a bmi category
# d = normalbmi.groupby(['common_user_id','bmicat']).agg({'bmi':['count']}).reset_index()
# d1 = d.groupby(['common_user_id']).agg({'bmicat':['count']}).reset_index()


# In[ ]:


# d.head()
# Counter(d.bmicat)
# d1.head()


# In[ ]:


# normid = d1[(d1['bmicat']['count'] == 1)].common_user_id


# In[ ]:


# normalfull = normalbmi[normalbmi.common_user_id.isin(normid)]


# In[ ]:


# normalfull = normalfull[normalfull.bmicat == 2]


# In[ ]:


# normalfull.head()


# In[ ]:


# #normalfull.to_pickle('/local/aling/normalfull.pkl')
# normalfull = pd.read_pickle(path +'/normalfull.pkl')


# In[ ]:


# Counter(normalfull.bmicat)


# In[ ]:


# normalfull.head()


# In[ ]:


# normalfull['value_grp'] = (normalfull.dt_diff.diff(1) != 0).cumsum()
# normal_stats = pd.DataFrame({'common_user_id':normalfull.groupby('value_grp').common_user_id.first(),
#                              'BeginDate':normalfull.groupby('value_grp').dt.first(),
#                              'EndDate':normalfull.groupby('value_grp').dt.last(),
#                              'bmi':normalfull.groupby('value_grp').bmi.first(),
#                              'bmicat':normalfull.groupby('value_grp').bmicat.first(),
#                              'dt_diff':normalfull.groupby('value_grp').dt_diff.sum(),
#                              'consecutive':normalfull.groupby('value_grp').size(),}).reset_index(drop=True)


# In[ ]:


# normal_stats.head()


# In[ ]:


# normal_stats['BeginDate'] = normal_stats['BeginDate'].dt.strftime('%Y-%m-%d')


# In[ ]:


# normal_stats.dtypes


# In[ ]:


# normal_stats.set_index('BeginDate', inplace = True)


# In[7]:


# normal_stats.head()
path = "/local/aling"
normal_stats = pd.read_pickle(path +'/normal_stats.pkl')


# In[8]:


normal_id = np.unique(normal_stats['common_user_id'])
norm_workouts = workouts[workouts.common_user_id.isin(normal_id)]
norm_workouts = norm_workouts[['common_user_id','time_taken','workout_date','calories_burned']]
norm_workouts['workout_date'] = pd.to_datetime(norm_workouts.workout_date, format = '%Y-%m-%d', errors = 'coerce')
norm_workouts['freq'] = pd.Series(np.ones([len(norm_workouts)]), index = norm_workouts.index)
normal_daily_workouts = norm_workouts.groupby(['common_user_id', pd.Grouper(key='workout_date', freq='D')])[['calories_burned','time_taken','freq']].sum().reset_index().sort_values('common_user_id')
normal_daily_workouts['workout_date'] = normal_daily_workouts['workout_date'].dt.strftime('%Y-%m-%d')
normal_daily_workouts.set_index('workout_date', inplace = True)


# In[ ]:


norm_workouts.dtypes


# In[10]:


normal_daily_workouts = normal_daily_workouts.reset_index()


# In[11]:


normal_daily_workouts = normal_daily_workouts.set_index(pd.DatetimeIndex(normal_daily_workouts['workout_date']))


# In[12]:


normal_daily_workouts.head()


# In[1]:


normal_daily_workouts.index[0]


# In[22]:


y.index[0]


# In[37]:


user_workouts = []
for j in range(0,len(y)-1):
    x = normal_daily_workouts.loc[y.index[j]:y.index[j+1]]
    if sum(x.common_user_id == normal_id[0]) == 0:
        df = pd.DataFrame({'common_user_id': normal_id[0],'time_taken':[0], 'calories_burned':[0], 'freq':[0]})
        agg_d =  {"calories_burned": ['mean'],
                "time_taken": ['mean'],
                "freq": ['sum']}
        f = df.groupby(["common_user_id"]).agg(agg_d).reset_index()
        addrow = f
    else:
        agg_d = {"calories_burned": ['mean'],
                "time_taken": ['mean'],
                "freq": ['sum']}
        f = x.groupby(["common_user_id"]).agg(agg_d).reset_index()
        addrow = pd.DataFrame(f.iloc[0]).T
    user_workouts.append(addrow)
user_workouts = pd.concat(user_workouts)


# In[38]:


user_workouts


# In[31]:


normal_daily_workouts.to_pickle('/local/aling/normal_daily_workouts.pkl')


# In[ ]:


# len(normal_id)


# In[ ]:


#normal_stats.to_pickle('/local/aling/normal_stats.pkl')


# In[ ]:


#find users who had low bmi at least once
s = weights2['bmicat'] == 1 #find all true 1's
s1 = weights2.common_user_id[s].reset_index(drop=True) #id numbers


# In[ ]:


len(s1)


# In[ ]:


lowbmi0 = weights2[weights2.common_user_id.isin(s1)]


# In[ ]:


lowbmi0.head()


# In[ ]:


#log everything per day
lowbmi = lowbmi0.groupby(['common_user_id', pd.Grouper(key='dt', freq='D')]).agg({'bmi': ['mean'],
     'bmicat':['first']}).reset_index().sort_values(['common_user_id','dt'])


# In[ ]:


lowbmi.head()


# In[ ]:


lowbmi['dt_diff'] = lowbmi.groupby('common_user_id')['dt'].apply(lambda x: x.diff()).dt.days
lowbmi['dt_diff'] = lowbmi['dt_diff'].fillna(0)


# In[ ]:


lowbmi.head()


# In[ ]:


Counter(lowbmi.bmicat)


# In[ ]:


#count number of times a user has been in a bmi category
d = lowbmi.groupby(['common_user_id','bmicat']).agg({'bmi':['count']}).reset_index()


# In[ ]:


d.head()


# In[ ]:


#filter based on user must have at least 2 logs of low bmi
r = d[(d['bmicat'] == 1) & (d['bmi']['count'] >= 2)]
r.head()


# In[ ]:


#recount number of times a user has been in a bmi category based on filter
r1 = r['common_user_id']
d[d.common_user_id.isin(r1)].head()


# In[ ]:


#get full stats of low bmi people
lowbmi_full = lowbmi[lowbmi.common_user_id.isin(r1)]


# In[ ]:


lowbmi_full.head()


# In[ ]:


print(len(lowbmi_full))
len(np.unique(lowbmi_full['common_user_id']))


# In[ ]:


lowbmi_full['value_grp'] = (lowbmi_full.bmicat.diff(1) != 0).astype('int').cumsum()


# In[ ]:


lowbmi_full.head()


# In[ ]:


lowbmi_stats = pd.DataFrame({'common_user_id':lowbmi_full.groupby('value_grp').common_user_id.first(),
                             'BeginDate':lowbmi_full.groupby('value_grp').dt.first(),
                             'EndDate':lowbmi_full.groupby('value_grp').dt.last(),
                             'bmi':lowbmi_full.groupby('value_grp').bmi.first(),
                             'bmicat':lowbmi_full.groupby('value_grp').bmicat.first(),
                             'dt_diff':lowbmi_full.groupby('value_grp').dt_diff.sum(),
                             'consecutive':lowbmi_full.groupby('value_grp').size(),}).reset_index(drop=True)


# In[ ]:


len(lowbmi_stats)


# In[ ]:


sum(lowbmi_stats.dt_diff >= 14)
final_lowbmi = lowbmi_stats[lowbmi_stats.dt_diff >= 14]


# In[ ]:


final_lowbmi['BeginDate'] = final_lowbmi['BeginDate'].dt.strftime('%Y-%m-%d')
final_lowbmi['EndDate'] = final_lowbmi['EndDate'].dt.strftime('%Y-%m-%d')


# In[ ]:


final_lowbmi.dtypes


# In[ ]:


print(len(np.unique(lowbmi_stats.common_user_id)))
print(len(np.unique(final_lowbmi.common_user_id)))


# In[ ]:


min(final_lowbmi.BeginDate)


# In[ ]:


max(final_lowbmi.EndDate)


# In[ ]:


#final_lowbmi.to_pickle('/local/aling/final_lowbmi.pkl')


# In[ ]:


#workouts = workouts[(workouts.workout_date >= '2014-03-03')&(workouts.workout_date <= '2017-04-21')]


# In[ ]:


#look at workouts and food entries on that date
#load in full workouts file
#start_time = time.time()
#workouts = pd.read_pickle('/local/aling/workouts.pkl')
#print("--- %s seconds ---" % (time.time() - start_time)) 


# In[ ]:


#low_workouts = workouts[workouts.common_user_id.isin(important_id)]


# In[ ]:


#low_workouts = low_workouts[['common_user_id','time_taken','workout_date','calories_burned']]


# In[ ]:


#len(low_workouts)


# In[ ]:


#low_workouts['workout_date'] = pd.to_datetime(low_workouts.workout_date, format = '%Y-%m-%d', errors = 'coerce')


# In[ ]:


#low_workouts['freq'] = pd.Series(np.ones([len(low_workouts)]), index = low_workouts.index)
#low_workouts.head()


# In[ ]:


#low_daily_workouts = low_workouts.groupby(['common_user_id', pd.Grouper(key='workout_date', freq='D')])[['calories_burned','time_taken','freq']].sum().reset_index().sort_values('common_user_id')
#low_daily_workouts.head()


# In[ ]:


#print(max(low_daily_workouts.workout_date))
#print(min(low_daily_workouts.workout_date))


# In[ ]:


#low_daily_workouts.to_pickle('/local/aling/low_daily_workouts.pkl')
low_daily_workouts.set_index('workout_date', inplace = True)


# In[ ]:


low_daily_workouts['workout_date'] = low_daily_workouts['workout_date'].dt.strftime('%Y-%m-%d')


# In[ ]:


low_daily_workouts.head()


# In[8]:


def squish_workouts(bmidf, workoutdf, important_id):
    user_workouts = []
    start_time = time.time()
    for i in range(0,len(important_id)):
        y = bmidf[bmidf.common_user_id == important_id[i]]
        df = pd.DataFrame({'common_user_id': important_id[i],'time_taken':[0], 'calories_burned':[0], 'freq':[0]})
        agg_d =  {"calories_burned": ['mean'],
                    "time_taken": ['mean'],
                    "freq": ['sum']}
        f = df.groupby(["common_user_id"]).agg(agg_d).reset_index()
        addrow = f
        user_workouts.append(addrow)
        for j in range(0,len(y)-1):
            x = workoutdf.loc[y.index[j]:y.index[j+1]]
            if sum(x.common_user_id == important_id[i]) == 0:
                df = pd.DataFrame({'common_user_id': important_id[i],'time_taken':[0], 'calories_burned':[0], 'freq':[0]})
                agg_d =  {"calories_burned": ['mean'],
                        "time_taken": ['mean'],
                        "freq": ['sum']}
                f = df.groupby(["common_user_id"]).agg(agg_d).reset_index()
                addrow = f
            else:
                agg_d = {"calories_burned": ['mean'],
                        "time_taken": ['mean'],
                        "freq": ['sum']}
                f = x.groupby(["common_user_id"]).agg(agg_d).reset_index()
                addrow = pd.DataFrame(f.iloc[0]).T
            user_workouts.append(addrow)
        print("--- %s seconds ---" % (time.time() - start_time))
    user_workouts = pd.concat(user_workouts)
    return user_workouts


# In[ ]:


#final_lowbmi.set_index('BeginDate', inplace = True)


# In[ ]:


final_lowbmi.head()


# In[9]:


important_id = np.unique(final_lowbmi['common_user_id'])
z = important_id[0:10]


# In[ ]:


squish_workouts(final_lowbmi, low_daily_workouts, z)


# In[ ]:


y =final_lowbmi[final_lowbmi.common_user_id == important_id[0]]
len(y)
final_lowbmi.head()


# In[ ]:


len(final_lowbmi)


# In[ ]:


low_daily_workouts.head()


# In[ ]:


for j in range(0,len(y)-1):
    print(y.index[j])
    print(y.index[j+1])


# In[ ]:


df = pd.DataFrame({'common_user_id': important_id[0],'time_taken':[0], 'calories_burned':[0], 'freq':[0]})
agg_d =  {"calories_burned": ['mean'],
                "time_taken": ['mean'],
                "freq": ['sum']}
f = df.groupby(["common_user_id"]).agg(agg_d).reset_index()
f


# In[ ]:


important_id = np.unique(final_lowbmi['common_user_id'])


# In[ ]:


x = low_daily_workouts.loc[y.index[0]:y.index[1]]
x.head()


# In[ ]:


agg_d = {
    "calories_burned": ['mean'],
    "time_taken": ['mean'],
    "freq": ['sum']
}
f = x.groupby(["common_user_id"]).agg(agg_d).reset_index()
t = pd.DataFrame(f.iloc[0]).T


# In[ ]:


added_df = pd.merge(final_lowbmi,low_daily_workouts, on = ['common_user_id'])


# In[ ]:


added_df.head()

