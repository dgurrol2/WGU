#!/usr/bin/env python
# coding: utf-8

# ## D4. 
# 
# Please see the treatment input code below.

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


#Generate histograms for variables with missing values and determine best method of treatment


# In[ ]:


#Generate distribution for Children variable 
plt.hist(df_churn['Children'])
plt.title("Children")


# In[ ]:


#Based on the skew of the distribution, the missing values will be imputed with the median of the Children variable
df_churn['Children'].fillna(df_churn['Children'].median(), inplace = True)


# In[ ]:


#Generate distribution for Age variable 
plt.hist(df_churn['Age'])
plt.title("Age")


# In[ ]:


#Based on the skew of the distribution, the missing values will be imputed with the rounded mean of the Age variable
df_churn['Age'].fillna(round(df_churn['Age'].mean()), inplace = True)


# In[ ]:


#Generate distribution for Income variable 
plt.hist(df_churn['Income'])
plt.title("Income")


# In[ ]:


#Based on the skew of the distribution, the missing values will be imputed with the median of the Income variable
df_churn['Income'].fillna(df_churn['Income'].median(), inplace = True)


# In[ ]:


#Generate distribution for Tenure variable 
plt.hist(df_churn['Tenure'])
plt.title("Tenure")


# In[ ]:


#Based on the skew of the distribution, the missing values will be imputed with the median of the Income variable
df_churn['Tenure'].fillna(df_churn['Tenure'].median(), inplace = True)


# In[ ]:


#Generate distribution for Bandwidth_GB_Year variable 
plt.hist(df_churn['Bandwidth_GB_Year'])
plt.title("Bandwidth_GB_Year")


# In[ ]:


#Based on the skew of the distribution, the missing values will be imputed with the median of the Income variable
df_churn['Bandwidth_GB_Year'].fillna(df_churn['Bandwidth_GB_Year'].median(), inplace = True)


# In[ ]:


#For categorical variables "Techie", "Phone" and "Income", impute with mode

df_churn['Techie'] = df_churn['Techie'].fillna(df_churn['Techie'].mode()[0])

df_churn['Phone'] = df_churn['Phone'].fillna(df_churn['Phone'].mode()[0])

df_churn['TechSupport'] = df_churn['TechSupport'].fillna(df_churn['TechSupport'].mode()[0])


# In[ ]:


#For quantitative variable "Outage_sec_perweek" impute outliers with median
df_churn["Outage_sec_perweek"] = np.where(df_churn["Outage_sec_perweek"] < 4, np.nan, df_churn["Outage_sec_perweek"])
df_churn["Outage_sec_perweek"] = np.where(df_churn["Outage_sec_perweek"] > 20, np.nan, df_churn["Outage_sec_perweek"])

#Impute NaN values with median
df_churn["Outage_sec_perweek"].fillna(df_churn["Outage_sec_perweek"].median(), inplace=True)


# In[ ]:


#Re-express categorical variables

#Step 1: Find unique values in Education column
print(df_churn['Education'].unique())

#Step 2: Create dictionary to ordinally encode Education values

dict_edu = {"Education": 
                {"Master's Degree": 18,
                        "Regular High School Diploma": 12,
                        "Doctorate Degree":24, 
                        "No Schooling Completed": 0,
                        "Associate's Degree": 15,
                        "Bachelor's Degree": 16,
                        "Some College, Less than 1 Year": 13,
                        "GED or Alternative Credential": 12,
                        "Some College, 1 or More Years, No Degree": 14,
                        "9th Grade to 12th Grade, No Diploma": 12,
                        "Nursery School to 8th Grade": 8,
                        "Professional School Degree": 20, 
                }
           }

#Step 3: Reexpress Education column as numeric
df_churn.replace(dict_edu, inplace = True)

#Step 4: Confirm values have been re-expressed
print(df_churn['Education'].unique())

