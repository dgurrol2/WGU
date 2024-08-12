#!/usr/bin/env python
# coding: utf-8

# ## D1. 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Please see the executable detection code below. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.decomposition import PCA


# In[ ]:


#Read and import 'churn_raw_data.csv' into to df_churn variable
df_churn = pd.read_csv('churn_raw_data.csv')


# In[ ]:


#detect duplicate rows in df_churn
print(df_churn.duplicated().value_counts())


# In[ ]:


#detect missing values in each variable of df_churn
df_churn.isnull().sum()


# In[ ]:


#Detect outliers in the following Quantitative variables using Boxplot() and Boxplot_info() functions
#Generate distrbutions for the following Quantitative variables using plt.hist()

#Population
#Children
#Age
#Income
#Outage_sec_perweek
#Email
#Contacts
#Yearly_equip_failure
#Tenure
#MonthlyCharge
#Bandwidth_GB_Year


# In[ ]:


#Create function to provide boxplot information
def boxplot_info(input):

    #obtain values of column and ignore nulls 
    data = input.dropna().values
    
    #generate q1 and q3 using pandas.DataFrame.quantile.  [In-text citation: (Pandas documentation)]
    q1 = input.quantile(0.25)
    print("Q1: " + str(q1))
    q3 = input.quantile(0.75)
    print("Q3: " + str(q3))
    
    #Calculate interquartile range for boxplot by subtracting Q1 from Q3
    iqr = q3 - q1
    print("IQR: " + str(iqr))
    
    #Calculate whisker values of boxplot. 
    whisker_lower = q1 - (1.5 * iqr)
    print("Lower Whisker: " + str(whisker_lower))
    whisker_upper = q3 + (1.5 * iqr)
    print("Upper Whisker: " + str(whisker_upper))
    
     #Find number of outliers outside of Q1 and Q3.  Print total number of outliers in column.  
    outliers_min = (input < whisker_lower).sum()
    print("Number of outliers lower than boxplot minimum: " + str(outliers_min))
    outliers_max = (input > whisker_upper).sum()
    print("Number of outliers greater than boxplot maximum: " + str(outliers_max))
    outliers_total = outliers_min + outliers_max
    print("Total number of Outliers: " + str(outliers_total))
    max_outlier = max(data)
    print("Highest Outlier: " + str(max_outlier))
    min_outlier = min(data)
    print("Lowest Outlier: " + str(min_outlier))


# In[ ]:


#Generate boxplot for Population variable
population_boxplot = sns.boxplot(x="Population", data = df_churn).set_title("Population")

#Generate boxplot info for Population using boxplot_info function
boxplot_info(df_churn['Population'])


# In[ ]:


#Generate distribution for Population variable 
plt.hist(df_churn['Population'])
plt.title("Population")


# In[ ]:


#Generate boxplot for Children variable
children_boxplot = sns.boxplot(x="Children", data = df_churn).set_title("Children")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Children'])


# In[ ]:


#Generate distribution for Children variable 
plt.hist(df_churn['Children'])
plt.title("Children")


# In[ ]:


#Generate boxplot for Age variable
age_boxplot = sns.boxplot(x="Age", data = df_churn).set_title("Age")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Age'])


# In[ ]:


#Generate distribution for Age variable 
plt.hist(df_churn['Age'])
plt.title("Age")


# In[ ]:


#Generate boxplot for Income variable
income_boxplot = sns.boxplot(x="Income", data = df_churn).set_title("Income")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Income'])


# In[ ]:


#Generate distribution for Income variable 
plt.hist(df_churn['Income'])
plt.title("Income")


# In[ ]:


#Generate boxplot for Outage_sec_perweek variable
outage_boxplot = sns.boxplot(x="Outage_sec_perweek", data = df_churn).set_title("Outage_sec_perweek")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Outage_sec_perweek'])


# In[ ]:


#Generate distribution for Outage_sec_perweek variable 
plt.hist(df_churn['Outage_sec_perweek'])
plt.title("Outage_sec_perweek")


# In[ ]:


#Generate boxplot for Email variable
email_boxplot = sns.boxplot(x="Email", data = df_churn).set_title("Email")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Email'])


# In[ ]:


#Generate distribution for Email variable 
plt.hist(df_churn['Email'])
plt.title("Email")


# In[ ]:


#Generate boxplot for Contacts variable
Contacts_boxplot = sns.boxplot(x="Contacts", data = df_churn).set_title("Contacts")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Contacts'])


# In[ ]:


#Generate distribution for Contacts variable 
plt.hist(df_churn['Contacts'])
plt.title("Contacts")


# In[ ]:


#Generate boxplot for Yearly_equip_failure variable
failure_boxplot = sns.boxplot(x="Yearly_equip_failure", data = df_churn).set_title("Yearly_equip_failure")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Yearly_equip_failure'])


# In[ ]:


#Generate distribution for Yearly_equip_failure variable 
plt.hist(df_churn['Yearly_equip_failure'])
plt.title("Yearly_equip_failure")


# In[ ]:


#Generate boxplot for Tenure variable
tenure_boxplot = sns.boxplot(x="Tenure", data = df_churn).set_title("Tenure")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Tenure'])


# In[ ]:


#Generate distribution for Tenure variable 
plt.hist(df_churn['Tenure'])
plt.title("Tenure")


# In[ ]:


#Generate boxplot for MonthlyCharge variable
MonthlyCharge_boxplot = sns.boxplot(x="MonthlyCharge", data = df_churn).set_title("MonthlyCharge")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['MonthlyCharge'])


# In[ ]:


#Generate distribution for MonthlyCharge variable 
plt.hist(df_churn['MonthlyCharge'])
plt.title("MonthlyCharge")


# In[ ]:


#Generate boxplot for Bandwidth_GB_Year variable
bandwidth_boxplot = sns.boxplot(x="Bandwidth_GB_Year", data = df_churn).set_title("Bandwidth_GB_Year")

#Generate boxplot info using boxplot_info() function
boxplot_info(df_churn['Bandwidth_GB_Year'])


# In[ ]:


#Generate distribution for Bandwidth_GB_Year variable 
plt.hist(df_churn['Bandwidth_GB_Year'])
plt.title("Bandwidth_GB_Year")


# In[ ]:




