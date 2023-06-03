#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('churn.csv')
data


# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data=data.drop('customerID',axis=1)


# In[6]:


data['Churn'].value_counts()


# In[7]:


sb.countplot(x='Churn',data=data)


# In[8]:


sb.countplot(x='gender',hue='Churn',data=data)


# In[9]:


sb.countplot(x='SeniorCitizen',hue='Churn',data=data)


# In[10]:


sb.countplot(x='Partner',hue='Churn',data=data)


# In[11]:


sb.countplot(x='Dependents',hue='Churn',data=data)


# In[12]:


sb.countplot(x='PhoneService',hue='Churn',data=data)


# In[13]:


sb.countplot(x='MultipleLines',hue='Churn',data=data)


# In[14]:


sb.countplot(x='InternetService',hue='Churn',data=data)


# In[15]:


sb.countplot(x='OnlineSecurity',hue='Churn',data=data)


# In[16]:


sb.countplot(x='DeviceProtection',hue='Churn',data=data)


# In[17]:


sb.countplot(x='TechSupport',hue='Churn',data=data)


# In[18]:


sb.countplot(x='StreamingTV',hue='Churn',data=data)


# In[19]:


sb.countplot(x='Contract',hue='Churn',data=data)


# In[20]:


sb.countplot(x='PaperlessBilling',hue='Churn',data=data)


# In[21]:


sb.countplot(x='PaymentMethod',hue='Churn',data=data)


# In[22]:


sb.countplot(x='PaymentMethod',hue='Churn',data=data)


# In[23]:


data.dtypes


# In[24]:


data=data.drop(columns=['gender','PhoneService','MultipleLines','InternetService','StreamingTV','StreamingMovies'],axis=1)


# In[25]:


print(data['TotalCharges'].dtypes)


# In[26]:


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')


# In[27]:


data=data.dropna(subset=['TotalCharges'])


# In[28]:


print(data['TotalCharges'].dtypes)


# In[29]:


data.isnull().sum()


# In[30]:


data=pd.get_dummies(data,columns=['SeniorCitizen','Partner','Dependents','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Contract','PaperlessBilling','PaymentMethod'])


# In[31]:


data['Churn']=data['Churn'].replace({'Yes':1,'No':0})


# In[32]:


from sklearn.preprocessing import LabelEncoder


# In[33]:


lb=LabelEncoder()


# In[34]:


#for column in data.columns:
#    if data[column].dtype==np.number:
#        continue
#    data[column]=lb.fit_transform(data[column])


# In[35]:


data


# In[36]:


plt.boxplot(data[['tenure','MonthlyCharges']])


# In[37]:


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[38]:


X=data.drop('Churn',axis=1)


# In[39]:


y=data['Churn']


# In[40]:


select=SelectKBest(chi2)


# In[41]:


select.fit(X,y)


# In[42]:


select.scores_


# In[43]:


score_col=pd.DataFrame({'Scores':select.scores_})


# In[44]:


col=pd.DataFrame({'Columns':X.columns})


# In[45]:


score_data=pd.concat([col,score_col],axis=1)


# In[46]:


score_data


# In[47]:


select_new=SelectKBest(score_func=f_classif)


# In[48]:


select_new.fit(X,y)


# In[49]:


scor_col1=select_new.scores_


# In[50]:


score_col_1=pd.DataFrame({'Scores':scor_col1})


# In[51]:


score_data_1=pd.concat([col,score_col_1],axis=1)


# In[52]:


score_data_1


# In[53]:


x=StandardScaler().fit_transform(X)


# In[57]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[58]:


lr=LogisticRegression()


# In[59]:


score=cross_val_score(lr,x,y,cv=10)


# In[60]:


score.mean()


# In[ ]:




