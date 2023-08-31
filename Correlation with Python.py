#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None


# In[2]:


#Read in data. Add 'r' if unicode error encountered

df = pd.read_csv(r"C:\Users\cecil\Downloads\movies.csv")


# In[3]:


#Explore data

df.head()


# In[4]:


df.info()


# In[5]:


#Find percentage of null values in each column

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[6]:


#Drop rows with Null values

df.dropna(how='any', axis=0, inplace = True)


# In[7]:


#Create new column for correct release year

df['correct_year'] = df['released'].astype(str).str.split(',').str[-1].astype(str).str[:5]


# In[8]:


#Order by gross rev.

df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[9]:


#Budget vs Gross scatter plt

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Film Budget')
plt.show()


# In[12]:


#Plot to show correlation

sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color':'red'}, line_kws={'color':'green'})


# In[20]:


#Correlation

df.corr()


# In[22]:


#Correlation Heatmap

correlation_matrix = df.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[24]:


#Correlation related to company

df_numerized = df

#Change object columns to numbers to see correlation
for col_name in df_numerized.columns:
    if (df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes 
    
df_numerized


# In[25]:


#Heatmap for all movie features

correlation_matrix = df_numerized.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[28]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[29]:


#Determine what movie features have highest correlation with gross revenue

high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[ ]:




