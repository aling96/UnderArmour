
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

import seaborn as sns
sns.set_style("whitegrid")

from scipy.stats import wilcoxon, mannwhitneyu, gaussian_kde
from collections import Counter, OrderedDict


# In[3]:


users = pd.read_csv("/remote/althoff/under_armour/derived_data/single_tables/users.csv", low_memory = False)
summary = pd.read_csv("/remote/althoff/under_armour/derived_data/single_tables/summary_table_20170309.csv", low_memory = False)


# In[3]:


users['created_at'] = pd.to_datetime(users.created_at, format = '%Y-%m-%d', errors = 'coerce')
users['last_login_at'] = pd.to_datetime(users.last_login_at, format = '%Y-%m-%d', errors = 'coerce')
users['len_use'] = ((users['last_login_at']-users['created_at']).dt.days)/365


# In[10]:


users.dtypes


# In[4]:


low = users.bmi < 18.5
normal = (users.bmi >= 18.5) & (users.bmi < 25)
low_normal = users.bmi < 25

low_id = users.common_user_id[low].reset_index(drop=True)
normal_id = users.common_user_id[normal].reset_index(drop=True)
low_normal_id = users.common_user_id[low_normal].reset_index(drop=True)


# In[5]:


low_normal_users = users[low_normal]
x = len(low_normal_users)


# In[58]:


x


# In[6]:


std = np.std(low_normal_users.goal_loss_per_week)
print(std)
print(std*3)


# In[7]:


low_normal_users.goal_loss_per_week.mean()


# In[8]:


sum(low_normal_users.goal_loss_per_week> 0.7315890788533005*3)


# In[65]:


#make bar plot of stats
df = pd.DataFrame({'characteristic': ['goal loss per week','workouts per day','weights per day'],
                  'val':[14844,3086,2607]})
ax = df.plot.bar(x='characteristic',y='val',rot=0)


# In[10]:


#number of weight measurement days
print(max(users.weight_measurement_days))
print(users.weight_measurement_days.mean())
#workout days
print(users.)


# In[11]:


min(users.age_years)


# In[12]:


Counter(users.gender)


# In[5]:


print(max(users.len_use))
print(min(users.len_use))
print(users.len_use.mean())


# In[6]:


plt.hist(users.len_use.dropna())
plt.show()


# In[7]:


plt.boxplot(users.len_use.dropna())
plt.show()


# In[8]:


low = summary.bmi < 18.5
normal = (summary.bmi >= 18.5) & (summary.bmi < 25)
over = (summary.bmi >= 25) & (summary.bmi < 30)
obese = (summary.bmi >= 30) 
total = len(summary)

lowid = summary[low].common_user_id
normid = summary[normal].common_user_id


# In[9]:


categories = [low,normal,over,obese]
catname = ['low','normal','over','obese']


# In[10]:


gender_table = np.zeros((5,2))


# In[11]:


for i in range(len(categories)):
    gender_table[i][0] = sum(summary.gender[categories[i]]== 'f')/sum(categories[i])
    gender_table[i][1] = sum(summary.gender[categories[i]] == 'm')/sum(categories[i])
gender_table[4][0] = sum(summary.gender == 'f')/total
gender_table[4][1] = sum(summary.gender == 'm')/total


# In[12]:


gender_df = pd.DataFrame(gender_table, columns = ['female','male'], index = ['low','normal','over','obese','all'])
gender_df


# In[9]:


def histogram_norm(data, bmi_cat, dataname, title, bminame,start,end,b):
    bins = np.linspace(start,end,b)
    for i in range(len(bmi_cat)):
        x = data[bmi_cat[i]]
        plt.hist(x, bins, alpha = 0.5, density='True')
    plt.title(title)
    plt.legend(bminame)
    plt.xlabel(dataname)
    plt.ylabel('Fraction of Users')
    plt.show()


# In[32]:


histogram_norm(low_normal_users.goal_loss_per_week, [low,normal], 'Goal Weightloss',
              'Distribution of Goal Weightloss by BMI', ['low','normal'],-2,3,20)


# In[14]:


def make_median_table(data, dim):
    table = np.zeros((5,dim))
    for i in range(len(categories)):
        table[i] = data[categories[i]].median()
    table[4] = data.median()
    return table


# In[15]:


#Counter(sorted(summary.age_years))


# In[16]:


realyears = summary.age_years[summary.age_years <= 111]
print(len(realyears))
print(total - len(realyears))
x = Counter(realyears)


# In[17]:


print(min(summary.age_years))
print(max(summary.age_years))


# In[61]:


min(summary[low].age_years)
min(summary[normal].age_years)
min(summary[over].age_years)
min(summary[obese].age_years)


# In[62]:


summary[low].age_years.mean()
summary[normal].age_years.mean()
summary[over].age_years.mean()
summary[obese].age_years.mean()


# In[18]:


#percent of ages less than 25 years old
(sum(summary.age_years <=25)/len(users))*100


# In[19]:


make_median_table(realyears, 1)


# In[49]:


#PDF for age
histogram_norm(summary.age_years, [low,normal,over, obese], 'Age','Distribution of Age by BMI', ['low','normal','over','obese'],0,100,100)


# In[21]:


females = summary[summary.gender == 'f']
males = summary[summary.gender == 'm']


# In[22]:


females_low = females[females.common_user_id.isin(lowid)]
females_norm = females[females.common_user_id.isin(normid)]
males_low = males[males.common_user_id.isin(lowid)]
males_norm = males[males.common_user_id.isin(normid)]


# In[23]:


def histogram_norm(data, bmi_cat, dataname, bminame,start,end,b):
    bins = np.linspace(start,end,b)
    for i in range(len(bmi_cat)):
        x = data[bmi_cat[i]]
        plt.hist(x, bins, alpha = 0.5, density='True')
    plt.title("Probability Density Function for " + dataname)
    plt.legend(bminame)
    plt.xlabel(dataname)
    plt.ylabel('Number')
    plt.show()


# In[50]:


plt.hist(females_low.age_years, bins = np.linspace(0,100,100), alpha = 0.5, density = 'True')
plt.hist(females_norm.age_years, bins = np.linspace(0,100,100),alpha = 0.5, density = 'True')
plt.hist(males_low.age_years, bins= np.linspace(0,100,100),alpha = 0.5, density = 'True')
plt.hist(males_norm.age_years, bins = np.linspace(0,100,100),alpha = 0.5, density = 'True')
plt.legend(['female_low','female_norm','male_low','male_norm'])
plt.title('Distribution of Age by Gender and BMI Category')
plt.xlabel('Age')
plt.ylabel('Fraction of Users')
plt.show()


# In[51]:


#geographic locations
c = Counter(summary.country)
print(len(c))
labels, values = zip(*Counter(sorted(summary.country)).items())
indexes = np.arange(len(labels))


# In[52]:


c_ordered = OrderedDict(c.most_common())
labelsc, valuesc = zip(*c_ordered.items())
indc = np.arange(len(labelsc))


# In[27]:


sum(valuesc)/total


# In[57]:


plt.figure(figsize=(20,10))
plt.bar(indc, valuesc)
plt.xticks(indc, labelsc, rotation = 90)
plt.show()


# In[29]:


totallogins = summary.total_logins.dropna()
print(max(summary.total_logins))
sub_totallogins = totallogins[totallogins <6000]
plt.hist(sub_totallogins)
plt.yscale('log')
plt.title('Total Logins')
print(len(sub_totallogins)/total)
print(len(sub_totallogins))


# In[30]:


total_logins_table = np.zeros((5,1))
for i in range(len(categories)):
    total_logins_table[i] = sub_totallogins[categories[i]].median()
total_logins_table[4] = sub_totallogins.median()
total_logins_table
make_median_table(sub_totallogins, 1)


# In[31]:


Counter(summary.total_login_days)
print(max(summary.total_login_days))
print(min(summary.total_login_days))
logindays = summary.total_login_days[summary.total_logins<6000]
len(logindays)
print(max(logindays))


# In[32]:


logindays_table = np.zeros((5,1))
for i in range(len(categories)):
    logindays_table[i] = logindays[categories[i]].median()
logindays_table[4] = logindays.median()
logindays_table


# In[33]:


plt.hist(logindays)


# In[34]:


histogram_norm(logindays, categories, 'Total Login Day', ['low','normal','over','obese'],0, 6000,100)


# In[35]:


logins_per_day = sub_totallogins/logindays
print(max(logins_per_day))
print(logins_per_day.mean())
print(logins_per_day.median())
print(len(logins_per_day))
print(len(logins_per_day[logins_per_day <10]))
len(logins_per_day[logins_per_day <10])/total


# In[36]:


logins_per_table = np.zeros((5,1))
for i in range(len(categories)):
    logins_per_table[i] = logins_per_day[categories[i]].median()
logins_per_table[4] = logins_per_day.median()
logins_per_table


# In[37]:


histogram_norm(logins_per_day, [obese,over,normal,low], 'Logins Per Day', ['obese','over','normal','low'],0, 10,100)


# In[38]:


Counter(summary.weight_measurement_days)
summary.weight_measurement_days.mean()
max(summary.weight_measurement_days)
#len(summary.weight_measurement_days)


# In[39]:


histogram_norm(summary.weight_measurement_days, categories, 'Weight Measurement Days', catname, 0,300,100)


# In[40]:


histogram_norm(summary.goal_loss_per_week, categories, 'Goal Loss Per Week', catname, -3,3,10)


# In[41]:


lowgoals = summary[low].goal_loss_per_week
normgoals = summary[normal].goal_loss_per_week


# In[42]:


plt.hist([lowgoals,normgoals], bins = np.linspace(-3,3,10))
plt.yscale('log')
plt.show()


# In[43]:


workouts_per_day = summary.n_workouts_daily_average.replace([np.inf, -np.inf], np.nan).dropna()
print(total-len(workouts_per_day))
print(len(workouts_per_day)/total)
print(max(workouts_per_day))
workouts_per_day = workouts_per_day[workouts_per_day < 40]


# In[44]:


x = Counter(workouts_per_day)
x.keys()
x.values()
plt.plot(x.keys(),x.values(), linestyle="",marker="o")
plt.title('Distribution workouts per day for all users')


# In[45]:


workouts_per_table = np.zeros((5,1))
for i in range(len(categories)):
    workouts_per_table[i] = workouts_per_day[categories[i]].median()
workouts_per_table[4] = workouts_per_day.median()
workouts_per_table


# In[46]:


plt.hist(workouts_per_day[workouts_per_day <40])
plt.yscale('log')
print(len(workouts_per_day[workouts_per_day <40])/len(workouts_per_day))


# In[47]:


histogram_norm(workouts_per_day, [obese,over,normal,low], 'Workouts Per Day', ['obese','over','normal','low'], 0,40,600)

