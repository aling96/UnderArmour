
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
import datetime
import glob
import os
import time

import seaborn as sns
sns.set_style("whitegrid")


from scipy.stats import wilcoxon, mannwhitneyu, expon, iqr
from scipy import stats
from collections import Counter, OrderedDict
import statsmodels.api as sm


# In[2]:


summary = pd.read_csv("/remote/althoff/under_armour/derived_data/single_tables/summary_table_20170309.csv", low_memory = False)


# In[4]:


summary.head().T


# In[4]:


low = summary.bmi < 18.5
normal = (summary.bmi >= 18.5) & (summary.bmi < 25)
over = (summary.bmi >= 25) & (summary.bmi < 30)
obese = (summary.bmi >= 30) 
total = len(summary)
low_normal = summary.bmi < 25

low_id = summary.common_user_id[low].reset_index(drop=True)
normal_id = summary.common_user_id[normal].reset_index(drop=True)
over_id = summary.common_user_id[over].reset_index(drop=True)
obese_id = summary.common_user_id[obese].reset_index(drop=True)

cat = [low_id,normal_id, over_id,obese_id]
name = ['low','normal','over','obese']
a = [1,.9,.8,.7]


# In[4]:


summary.columns


# In[39]:


plt.hist(summary.n_workouts_daily_average.dropna(), bins = np.linspace(0,5), density=True)
plt.show()


# In[3]:


#average food entry days
summary.total_food_entry_days.mean()


# In[22]:


#average length recording workouts
summary.workouts_period_days.mean()


# In[16]:


#average total workout days recorded
summary.workout_date_count.mean()


# In[5]:


#average length of use of app
print(summary.total_login_days.mean())
print(summary.total_login_days.mean()/365)


# In[21]:


(pd.to_datetime(summary.last_login_at) - pd.to_datetime(summary.created_at)).mean()


# In[6]:


def make_table(data, categories, variable,stats):
    table = np.zeros((5,len(stats)*2))
    for i in range(len(categories)):
        for j in range(len(stats)):
            table[i][j] = data[data.common_user_id.isin(categories[i])][variable][stats[j]].median()
            table[i][j+3] = data[data.common_user_id.isin(categories[i])][variable][stats[j]].mean()
    for j in range(len(stats)):
        table[4][j] = data[variable][stats[j]].median()
        table[4][j+3] = data[variable][stats[j]].mean()
    return table


# In[2]:


#load in full workouts file
path = "/local/aling"
start_time = time.time()
df = pd.read_pickle(path+'/workouts.pkl')
print("--- %s seconds ---" % (time.time() - start_time)) 


# In[9]:


#print(len(df)) = 169,265,594
#print(len(np.unique(df.common_user_id))) =1,709,223


# In[3]:


df['workout_date'] = pd.to_datetime(df.workout_date, format = '%Y-%m-%d', errors = 'coerce')
df['freq'] = pd.Series(np.ones([len(df)]), index = df.index)


# In[6]:


#df.head().T


# In[5]:


#workout data frame that counts by day
start_time = time.time()
workouts_per_day = df.groupby(['common_user_id', pd.Grouper(key='workout_date', freq='D')])[['calories_burned','time_taken','freq']].sum().reset_index().sort_values('workout_date')
print("--- %s seconds ---" % (time.time() - start_time)) 
workouts_per_day.to_pickle('/local/aling/workouts_per_day.pkl')


# In[ ]:


#workout data frame that counts by month
start_time = time.time()
month_df = df.groupby(['common_user_id', pd.Grouper(key='workout_date', freq='M')])[['calories_burned','time_taken','freq']].sum().reset_index().sort_values('workout_date')
print("--- %s seconds ---" % (time.time() - start_time)) 
month_df.to_pickle('/local/aling/full_month_df.pkl')


# In[5]:


#start with this part
start_time = time.time()
workouts_per_day = pd.read_pickle('/local/aling/workouts_per_day.pkl')
print("--- %s seconds ---" % (time.time() - start_time)) 


# In[6]:


low_normal_id = summary.common_user_id[low_normal].reset_index(drop=True)


# In[7]:


workouts_per_day_nl= workouts_per_day[workouts_per_day.common_user_id.isin(low_normal_id)]


# In[8]:


workouts_per_day_nl.head()


# In[49]:


avg_workout_df = workouts_per_day_nl.groupby('common_user_id').mean()


# In[50]:


avg_workout_df.time_taken = avg_workout_df.time_taken/3600


# In[11]:


avg_workout_df.head()


# In[12]:


total_people = len(avg_workout_df)
total_people


# In[13]:


#time taken avg
plt.hist(avg_workout_df.time_taken, bins= np.linspace(0,6))
plt.title('Distribution of Hours of Workout per day')
plt.ylabel('Fraction of Users')
plt.xlabel('Hours per day')
plt.show()


# In[63]:


max(avg_workout_df.time_taken)


# In[97]:


avg_workout_df= avg_workout_df[(avg_workout_df.time_taken < 24) & 
                              (avg_workout_df.time_taken >=0.5)]


# In[90]:


print(avg_workout_df.time_taken.mean())
print(avg_workout_df.time_taken.median())
min(avg_workout_df.time_taken)


# In[94]:


sd_time = np.std(avg_workout_df[avg_workout_df.time_taken < 24].time_taken)
sd_time*3


# In[93]:


#people >=3 std above mean for time exercising per day
sum(avg_workout_df.time_taken > sd_time*3)


# In[98]:


#workouts per day
plt.hist(avg_workout_df.freq, bins= np.linspace(0,4), density=True)
plt.title('Distribution of Number of Workouts per day')
plt.ylabel('Fraction of Users')
plt.xlabel('Number of Workouts')
plt.show()


# In[46]:


avg_workout_df.freq = round(avg_workout_df.freq)


# In[99]:


avg_workout_df.freq.mean()


# In[100]:


sd_freq = np.std(avg_workout_df.freq)
sd_freq


# In[101]:


sd_freq*3


# In[102]:


#people >=3 std above mean for workouts per day
sum(avg_workout_df.freq > round(sd_freq)*3)


# In[61]:


outlierworkout = avg_workout_df[avg_workout_df.freq >3]


# In[20]:


print(len(month_df))
month_df.head()


# In[15]:


month_df.dtypes


# In[7]:


month_df['time_taken'] = month_df['time_taken'].div(3600)


# In[17]:


month_df.head()


# In[8]:


month_df1 = month_df[(month_df['freq'] >=2) & (month_df['freq'] <=1000)]
month_df1 = month_df1[(month_df1['time_taken'] >0) & (month_df1['time_taken'] <=730)]
#because there are 730 hours in one month
print(len(month_df1))


# In[5]:


#month_df1.to_pickle('/local/aling/month_workouts.pkl')
path = "/local/aling"
os.chdir(path)
start_time = time.time()
month_df1 = pd.read_pickle('./month_workouts.pkl')
print("--- %s seconds ---" % (time.time() - start_time)) 


# In[22]:


len(month_df1)


# In[28]:


a = month_df1[month_df1.common_user_id.isin(low_id)]
a = a.replace({'bmicat': 0}, 1)
b = month_df1[month_df1.common_user_id.isin(normal_id)]
b = b.replace({'bmicat': 0}, 2)
c = month_df1[month_df1.common_user_id.isin(over_id)]
c = c.replace({'bmicat': 0}, 3)
d = month_df1[month_df1.common_user_id.isin(obese_id)]
d = d.replace({'bmicat': 0}, 4)


# In[36]:


month_df2 = pd.concat([a,b,c,d], ignore_index = True, axis = 0)
month_df2 = month_df2.sort_values('common_user_id')


# In[38]:


month_df2  = month_df2.rename(columns={"time_taken": "exercise_hours", "freq": "workout_logs"})


# In[41]:


workout_logs = month_df2[['common_user_id','workout_logs','bmicat']]


# In[44]:


workout_logs.to_csv('/local/aling/workout_logs.csv')


# In[9]:


print(month_df1['time_taken'].median())
print(month_df1['freq'].median())


# In[10]:


print(max(month_df1['time_taken']))
print(max(month_df1['freq']))


# In[11]:


print(min(month_df1['time_taken']))
print(min(month_df1['freq']))


# In[7]:


def histogram_norm(data,variable, stat, bmi_cat, bminame,start,end,b):
    bins = np.linspace(start,end,b)
    for i in range(len(bmi_cat)):
        x = data[data.common_user_id.isin(bmi_cat[i])]
        if stat == 'n':
            plt.hist(x[variable], bins, alpha = a[i], density='True')
        else:
            plt.hist(x[variable][stat], bins, alpha = a[i], density='True')
    plt.title("Distribution of Workout Logs per Month")
    plt.legend(bminame)
    plt.ylim(0,.14)
    plt.xlabel('Number of Workouts Logs')
    plt.ylabel('Fraction of Users')
    plt.show()


# In[89]:


#histogram_norm(month_df, 'freq','n', cat, "Month Frequency",'count per month',name,0,80,40,'l')
histogram_norm(month_df1, 'freq','n', cat,name,0,80,40)


# In[13]:


bins = np.linspace(0,80,40)
x = month_df1[month_df1.common_user_id.isin(obese_id)]
plt.hist(x['freq'], bins,color='red', density='True')
plt.ylim(0,.14)
plt.title("Distribution of Workouts Logs per Month")
plt.xlabel('Number of Workouts Logs')
plt.ylabel('Fraction of Users')
plt.show()


# In[25]:


agg_d = {
    "calories_burned": ["sum","median","mean"],
    "time_taken": ["sum","median","mean","count"],
}
intensity_month = month_df.groupby("common_user_id").agg(agg_d).reset_index()


# In[16]:


agg_d = {
    "calories_burned": ["sum","median","mean"],
    "time_taken": ["sum","median","mean","count"],
}
intensity_month1 = month_df1.groupby("common_user_id").agg(agg_d).reset_index()


# In[27]:


intensity_month.head()


# In[69]:


#histogram_norm(intensity_month1,'time_taken','sum',cat, "Filtered Time Taken Sum by Month",'hours',name,0,500,50,'l')


# In[92]:


histogram_norm(intensity_month1,'time_taken','mean',cat, name,0,50,50)


# In[70]:


#histogram_norm(intensity_month1,'time_taken','median',cat, "Filtered Time Taken Median by Month",'hours',name,0,50,50,'l')


# In[19]:


def expon_fit(data, var, stat, bmicat, start,end,space):
    bmimean = np.zeros((4,3))
    for i in range(len(cat)): 
        x = data[data.common_user_id.isin(bmicat[i])]
        if stat == 'n':
            x1 = x[var]
        else:
            x1 = x[var][stat]
        loc, scale = expon.fit(x1)
        f = np.linspace(start,end,space)
        y = expon.pdf(f,loc,scale)
        plt.plot(f,y)
        plt.ylim(0,.14)
        bmimean[i,0] = x1.mean()
        bmimean[i,1] = x1.median()
        bmimean[i,2] = x1.std()
    plt.show()
    return(bmimean)


# In[32]:


bmimean = np.zeros((4,3))


# In[ ]:


expon_fit(intensity_month1, 'time_taken','median',cat,0,50,50)


# In[18]:


meanbmistat = expon_fit(intensity_month1, 'time_taken','mean',cat,0,50,50)
bmis = ['Low','Normal','High','Obese']
xpos = np.arange(len(bmis))
timemeans = meanbmistat[:,0]
timestds = meanbmistat[:,2]
fig,ax = plt.subplots()
ax.bar(xpos, timemeans, yerr = timestds, align = 'center', color=['blue', 'orange','green','red'],
       ecolor = 'black', capsize = 10)
ax.set_ylabel('Mean Hours of Exercise')
ax.set_xticks(xpos)
ax.set_xticklabels(bmis)
ax.set_title('Mean Hours of Exercise per Month by BMI')
ax.yaxis.grid(True)
plt.ylim(0,30)
plt.tight_layout()
plt.show()


# In[ ]:


expon_fit(intensity_month1, 'time_taken','sum',cat,0,500,50)


# In[20]:


freqstat = expon_fit(month_df1, 'freq','n',cat,0,80,40)
bmis = ['Low','Normal','High','Obese']
xpos = np.arange(len(bmis))
freqmeans = freqstat[:,0]
freqstds = freqstat[:,2]

plt.bar(xpos, freqmeans, yerr = freqstds, align = 'center', color=['blue', 'orange','green','red'],
       ecolor = 'black', capsize = 10)
plt.ylabel('Number of Workout Logs')
plt.xticks(xpos, bmis)
plt.title('Number of Workout Logs per Month by BMI')
plt.ylim(0,30)
plt.tight_layout()
plt.show()


# In[ ]:


def test_sigdiff(data, var, stat, bmi1, bmi2, a):
    x = data[data.common_user_id.isin(bmi1)]
    if a == 'a':
        y = data[~data.common_user_id.isin(bmi1)]
        print("Variables: low bmi and all other bmi's")
    else:
        y = data[data.common_user_id.isin(bmi2)]
    if stat == 'n':
        x1 = x[var]
        y1 = y[var]
        print("Significance Tests for " + var)
    else:
        x1 = x[var][stat]
        y1 = y[var][stat]
        print("Significance Tests for " + stat)
    locx, scalex = expon.fit(x1)
    locy, scaley = expon.fit(y1)
    rvsx = expon.rvs(locx, scalex, size = 300)
    rvsy = expon.rvs(locy, scaley, size = 300)
     
    print(str(stats.ks_2samp(rvsx, rvsy)))
    print(str(stats.ttest_ind(rvsx,rvsy, equal_var = False)))


# In[ ]:


test_sigdiff(intensity_month1, 'time_taken','median', low_id, low_id, 'a')


# In[ ]:


test_sigdiff(intensity_month1, 'time_taken','mean', low_id, low_id, 'a')


# In[ ]:


test_sigdiff(intensity_month1, 'time_taken','sum', low_id, low_id, 'a')


# In[ ]:


test_sigdiff(month_df1, 'freq','n', low_id, low_id, 'a')


# In[ ]:


c = [normal_id, over_id, obese_id]
for b in c:
    test_sigdiff(month_df1,'freq','n',low_id,b,'x')


# In[ ]:


c = [normal_id, over_id, obese_id]
for b in c:
    test_sigdiff(intensity_month1, 'time_taken','mean', low_id,b,'x')


# In[ ]:


c = [normal_id, over_id, obese_id]
for b in c:
    test_sigdiff(intensity_month1, 'time_taken','median', low_id,b,'x')


# In[ ]:


c = [normal_id, over_id, obese_id]
for b in c:
    test_sigdiff(intensity_month1, 'time_taken','sum', low_id,b,'x')

