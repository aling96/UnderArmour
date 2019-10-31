
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

from scipy.stats import wilcoxon, mannwhitneyu, expon
from scipy import stats
import collections
import statsmodels.api as sm


# In[2]:


path = "/local/aling"
start_time = time.time()
users1 = pd.read_pickle(path +'/users1.pkl')
print("--- %s seconds ---" % (time.time() - start_time))


# In[3]:


start_time = time.time()
workouts = pd.read_pickle('/local/aling/month_workouts.pkl')
print("--- %s seconds ---" % (time.time() - start_time)) 


# In[4]:


path = "/local/aling"
start_time = time.time()
food = pd.read_pickle(path +'/food1_day.pkl')
print("--- %s seconds ---" % (time.time() - start_time))


# In[5]:


low = users1.bmi_start < 18.5


# In[6]:


len(users1)


# In[7]:


#goal loss per week
print('goal loss per week stats')
print(users1.goal_loss_per_week.mean())
print(users1.goal_loss_per_week.median())
print(max(users1.goal_loss_per_week))
print(min(users1.goal_loss_per_week))
#total weight loss
print('total weight loss stats')
print(max(users1.total_weight_loss))
print(min(users1.total_weight_loss))
print(users1.total_weight_loss.mean())
print(users1.total_weight_loss.median())


# In[8]:


#split low bmi by goal loss per week
lowbmistart = users1[low]
lowbmistart.head()
print(len(lowbmistart))


# In[9]:


gain = lowbmistart['goal_loss_per_week'] <0
lose = (lowbmistart['goal_loss_per_week'] >=0)
#same = (lowbmistart['goal_loss_per_week'] == 0)

gain_id = lowbmistart.common_user_id[gain].reset_index(drop=True)
lose_id = lowbmistart.common_user_id[lose].reset_index(drop=True)
#same_id = lowbmistart.common_user_id[same].reset_index(drop=True)

cat = [ lose_id,gain_id]
name = ['lose','gain']


# In[28]:


plt.hist(lowbmistart['goal_loss_per_week'])
plt.title('Low BMI goal loss per week')
plt.xlabel('lbs')
plt.show()

print(str(sum(lose)) + " wants to lose weight" )
#print(str(sum(same)) + " wants to stay the same" )
print(str(sum(gain)) + " wants to gain weight" )


# In[11]:


plt.hist(lowbmistart['total_weight_loss'])
plt.title('Low BMI total weight loss')
plt.xlabel('lbs')
plt.show()
print(str(sum(lowbmistart['total_weight_loss'] >0)) + " lost weight" )
print(str(sum(lowbmistart['total_weight_loss'] == 0)) + " stayed the same" )
print(str(sum(lowbmistart['total_weight_loss'] < 0)) + " gained weight" )


# In[29]:


def histogram_norm(data,variable, stat, bmi_cat, bminame,start,end,b):
    bins = np.linspace(start,end,b)
    for i in range(len(bmi_cat)):
        x = data[data.common_user_id.isin(bmi_cat[i])]
        if stat == 'n':
            plt.hist(x[variable], bins, alpha = .8, density='True')
        else:
            plt.hist(x[variable][stat], bins, alpha = .8, density='True')
    plt.title("Distribution of Exercise per Month")
    plt.legend(bminame)
    plt.ylim(0,0.18)
    plt.xlabel('Number of Workout Logs')
    plt.ylabel('Fraction of Users')
    plt.show()


# In[14]:


def test_sigdiff(data, var, stat, bmi1, bmi2):
    x = data[data.common_user_id.isin(bmi1)]
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


# In[15]:


food.head()


# In[16]:


histogram_norm(food, 'total_calories','count', cat, "Daily Food Entries",'entries',name,0,20,20,'l')
test_sigdiff(food,'total_calories','count',lose_id, gain_id)
expon_fit(food, 'total_calories','count',cat,0,20,20)


# In[17]:


histogram_norm(food, 'total_calories','sum', cat, "Daily Calories >= 500",'calories',name,400,3000,20,'l')
test_sigdiff(food,'total_calories','sum',lose_id, gain_id)
expon_fit(food, 'total_calories','sum',cat,400,3000,20)


# In[18]:


workouts.head()


# In[19]:


agg_d = {
    "calories_burned": ["sum","median","mean"],
    "time_taken": ["sum","median","mean","count"],
}
intensity_month1 = workouts.groupby("common_user_id").agg(agg_d).reset_index()


# In[35]:


def expon_fit(data, var, stat, bmicat, start,end,space):
    bmimean = np.zeros((2,3))
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
        plt.ylim(0,.18)
        bmimean[i,0] = x1.mean()
        bmimean[i,1] = x1.median()
        bmimean[i,2] = x1.std()
    plt.show()
    return(bmimean)


# In[40]:


histogram_norm(workouts, 'freq','n', cat, name,0,80,40)
meanbmistat = expon_fit(workouts, 'freq','n',cat,0,80,40)
bmis = ['Lose weight','Gain weight']
xpos = np.arange(len(bmis))
timemeans = meanbmistat[:,0]
timestds = meanbmistat[:,2]
fig,ax = plt.subplots()
ax.bar(xpos, timemeans, yerr = timestds, align = 'center', color=['blue', 'orange','green','red'],
       ecolor = 'black', capsize = 10)
ax.set_ylabel('Mean Workouts Logged')
ax.set_xticks(xpos)
ax.set_xticklabels(bmis)
ax.set_title('Mean Workouts Logged per Month by Weightloss Goal')
ax.yaxis.grid(True)
plt.tight_layout()
plt.ylim(0,30)
plt.show()
#test_sigdiff(workouts,'freq','n',lose_id, gain_id)
#expon_fit(workouts, 'freq','n',cat,0,80,40)


# In[21]:


histogram_norm(intensity_month1,'time_taken','sum',cat, "Low BMI Workout Time Taken Sum by Month",'hours',name,0,500,50,'l')
test_sigdiff(intensity_month1,'time_taken','sum',lose_id, gain_id)
expon_fit(intensity_month1, 'time_taken','sum',cat,0,500,50)


# In[33]:


histogram_norm(intensity_month1,'time_taken','mean',cat, name,0,50,50)
meanbmistat = expon_fit(intensity_month1, 'time_taken','mean',cat,0,80,40)
bmis = ['Lose weight','Gain weight']
xpos = np.arange(len(bmis))
timemeans = meanbmistat[:,0]
timestds = meanbmistat[:,2]
fig,ax = plt.subplots()
ax.bar(xpos, timemeans, yerr = timestds, align = 'center', color=['blue', 'orange'],ecolor = 'black', capsize = 10)
ax.set_ylabel('Mean Hours of Exercise')
ax.set_xticks(xpos)
ax.set_xticklabels(bmis)
ax.set_title('Mean Hours of Exercise per Month by BMI')
ax.yaxis.grid(True)
plt.ylim(0,30)
plt.tight_layout()
plt.show()


# In[34]:


timemeans


# In[23]:


histogram_norm(intensity_month1,'time_taken','median',cat, "Low BMI Workout Time Taken Median by Month",'hours',name,0,50,50,'l')
test_sigdiff(intensity_month1,'time_taken','median',lose_id, gain_id)
expon_fit(intensity_month1, 'time_taken','median',cat,0,50,50)

