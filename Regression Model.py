
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
from random import sample
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


# In[2]:


path = "/local/aling"
testweights2 = pd.read_pickle(path +'/testweightsbiweek.pkl')
controlweights2 = pd.read_pickle(path +'/controlweightsbiweek.pkl')

test_weight_workouts = pd.read_pickle('/local/aling/test_weight_workouts.pkl')
control_weight_workouts = pd.read_pickle('/local/aling/control_weight_workouts.pkl')


# In[5]:


test_weight_workouts.head()


# In[6]:


control_weight_workouts.head()


# In[7]:


controldf1 = control_weight_workouts[['bmicat','time_taken','freq','calories_burned']]
controldf1.head()


# In[8]:


testdf1 = test_weight_workouts[['bmicat','time_taken','freq','calories_burned']]
testdf1.head()


# In[24]:


fulldf = testdf1.append(controldf1)
fulldf.loc[fulldf['bmicat'] ==1, 'bmicat'] = 0
fulldf.loc[fulldf['bmicat'] ==2, 'bmicat'] = 1
fulldf.head()


# In[25]:


Counter(fulldf.bmicat)


# In[26]:


arrayframe = fulldf.values
arrayframe


# In[19]:


arrayframe[:,0]


# In[27]:


X = arrayframe[:,1:4]
X = X.astype('int')
Y = arrayframe[:,0]
Y = Y.astype('int')


# In[28]:


X[0]


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[30]:


logreg = LogisticRegressionCV(cv = 10,random_state=0, solver = 'lbfgs', multi_class = 'multinomial', max_iter=10000)


# In[31]:


logreg.fit(X_train,y_train)


# In[32]:


print(logreg.coef_)
logreg.fit(X / np.std(X, 0), Y)


# In[33]:


y_pred = logreg.predict(X_test)


# In[34]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[35]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[36]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[37]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[38]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


#print type(newY)# pandas core frame
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)
print(len(X_test))
print(len(y_test))


# In[39]:


from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha = 0.3, normalize = True)
lassoReg.fit(X_train,y_train)
pred = lassoReg.predict(X_test)
mse = np.mean((pred - y_test)**2)
print(lassoReg.score(X_test,y_test))
print(mse)
x_plot = plt.scatter(pred,(pred-y_test), c='b')
plt.hlines(y=0,xmin = -1000, xmax = 5000)
plt.title('Residual plot')
#coef = Series(lreg.coef_,predictors).sort_values()
#coef.plot(kind='bar', title='Modal Coefficients')


# In[40]:


lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)
train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)


# In[41]:


print("linear regression train score: " + str(train_score))
print("linear regression test score: " +  str(test_score))
print("ridge regression train score low alpha: " + str(Ridge_train_score))
print("ridge regression test score low alpha: "+ str(Ridge_test_score))
print("ridge regression train score high alpha: " + str(Ridge_train_score100))
print("ridge regression test score high alpha: " + str(Ridge_test_score100))


# In[42]:


plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()


# In[43]:


from sklearn.linear_model import Lasso
X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.3, random_state=31)
lasso = Lasso()
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)


# In[44]:


print(train_score)
print(test_score)
print(coeff_used)


# In[45]:


lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,y_train)


# In[46]:


train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)


# In[47]:


print(train_score001)
print(test_score001)
print(coeff_used001)


# In[48]:


lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)


# In[49]:


print(train_score00001)
print(test_score00001)
print(coeff_used00001)


# In[50]:


lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print(lr_train_score)
print(lr_test_score)


# In[51]:



plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)

plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()

