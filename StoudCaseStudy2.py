#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd
import numpy as np


# In[266]:


df = pd.read_csv('casestudy.csv',usecols=['customer_email','net_revenue','year'])


# In[267]:


df['year'] = pd.to_datetime(df['year'],format='%Y').dt.year


# In[268]:


df['customer_email'] = df['customer_email'].str.rstrip()#remve empty strings
df['customer_email'] = df['customer_email'].str.lstrip()#remove empty string
df['customer_email'] = df['customer_email'].str.lower()


# In[269]:


df.head()


# ## Let's start answering the questions asked 

# ### 1. Total revenue current year 

# In[270]:


print("Revenue for the year \033[1m 2017 \033[0m is: ${0}".format( round(df[df['year']==2017]['net_revenue'].sum()),3) )


# ### 2. New Customer revenue 

# In[271]:


in_2016_not_2015 = list(set(df[df['year']==2016]['customer_email']) - set(df[df['year']==2015]['customer_email']))
in_2017_not_2016 = list(set(df[df['year']==2017]['customer_email']) - set(df[df['year']==2016]['customer_email']))


# In[272]:


new_customer_revenue = df[(df['customer_email'].isin(in_2016_not_2015))&(df['year']==2016)]['net_revenue'].sum() + df[(df['customer_email'].isin(in_2017_not_2016))&(df['year']==2017)]['net_revenue'].sum()


# In[273]:


print("New customer revenue for customers who joined in \033[1m2016\033[0m and not in \033[1m2015\033[0m and who joined in \033[1m2017\033[0m and were not in \033[1m2016\033[0m is:\n $",round(new_customer_revenue,3))


# ### 3. Existing customer growth 

# In[274]:


in_2016_and_2015 = list( set(df[df['year']==2016]['customer_email']).intersection(set(df[df['year']==2015]['customer_email'])) )
in_2017_and_2016 = list( set(df[df['year']==2017]['customer_email']).intersection(set(df[df['year']==2016]['customer_email'])) )


# In[275]:


existing_customer_revenue = df[(df['customer_email'].isin(in_2017_and_2016))&(df['year']==2017)]['net_revenue'].sum() -  df[(df['customer_email'].isin(in_2016_and_2015))&(df['year']==2016)]['net_revenue'].sum()


# In[276]:


print("The Existing customer growth if we compare the \033[1m2017\033[0m existing customer growth and \033[1m2016\033[0m existing customer growth: $",round(existing_customer_revenue,3))


# ### 4. Revenue lost from attrition 

# In[277]:


not_2016_in_2015 = list( set(df[df['year']==2015]['customer_email']) - set(df[df['year']==2016]['customer_email']) )
not_2017_in_2016 = list( set(df[df['year']==2016]['customer_email']) - set(df[df['year']==2017]['customer_email']) )


# In[278]:


revenue_lost_attrition = df[(df['customer_email'].isin(not_2016_in_2015))&(df['year']==2015)]['net_revenue'].sum() + df[(df['customer_email'].isin(not_2017_in_2016))&(df['year']==2016)]['net_revenue'].sum()


# In[279]:


print('Revenue lost due to attrition: $',round(revenue_lost_attrition,3))


# ### 5. Existing customer revenue current year

# In[280]:


in_2017_and_2016 = list( set(df[df['year']==2017]['customer_email']).intersection(set(df[df['year']==2016]['customer_email'])) )


# In[281]:


existing_cus_revenue = df[(df['customer_email'].isin(in_2017_and_2016))&(df['year']==2017)]['net_revenue'].sum()


# In[282]:


print('Existing Customer Revenue for current year \033[1m2017\033[0m : $',round(existing_cus_revenue,3))


# ### 6. Existing Customer Revenue Prior Year

# In[283]:


in_2016_and_2015 = list( set(df[df['year']==2016]['customer_email']).intersection(set(df[df['year']==2015]['customer_email'])) )


# In[284]:


existing_cus_revenue_prior = df[(df['customer_email'].isin(in_2016_and_2015))&(df['year']==2016)]['net_revenue'].sum()


# In[285]:


print('Existing Customer Revenue for prior year \033[1m2016\033[0m: $',round(existing_cus_revenue_prior,3))


# ### 7. Total Customers Current Year

# In[286]:


total_customer_current_year = df[df['year']==2017]['customer_email'].nunique()


# In[287]:


print('Total customers current year \033[1m2017\033[0m is : ',total_customer_current_year)


# ### 8. Total Customers Previous year 

# In[288]:


total_customer_previous_year = df[df['year']==2016]['customer_email'].nunique()


# In[289]:


print('Total customers previous year \033[1m2016\033[0m is : ',total_customer_previous_year)


# ### 9. New Customers 

# In[290]:


in_2016_not_2015 = list(set(df[df['year']==2016]['customer_email']) - set(df[df['year']==2015]['customer_email']))
in_2017_not_2016 = list(set(df[df['year']==2017]['customer_email']) - set(df[df['year']==2016]['customer_email']))


# In[291]:


total_new_customers = len(in_2016_not_2015)+len(in_2017_not_2016)


# In[292]:


print('Total new customers gained in year \033[1m2016\033[0m and \033[1m2017\033[0m :',total_new_customers)


# ### 10. Lost Customers 

# In[293]:


not_2016_in_2015 = list( set(df[df['year']==2015]['customer_email']) - set(df[df['year']==2016]['customer_email']) )
not_2017_in_2016 = list( set(df[df['year']==2016]['customer_email']) - set(df[df['year']==2017]['customer_email']) )


# In[294]:


total_lost_customers = len(not_2016_in_2015) + len(not_2017_in_2016)


# In[295]:


print('Total customers lost that were in \033[1m2015\033[0m and not in \033[1m2016\033[0m and with the company in \033[1m2016\033[0m and not in \033[1m2017\033[0m : ',total_lost_customers)


# # Visualization 

# In[296]:


import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.graph_objects as go #designing in plotly
# import chart_studio.plotly as pt
from plotly.subplots import make_subplots #add_subplots
import plotly.express as px #to work with tidy objects
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.offline import init_notebook_mode,iplot,plot,download_plotlyjs
init_notebook_mode()
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[297]:


ax = sns.barplot(data=df,x=[2015,2016,2017],y=df.groupby('year')['net_revenue'].sum().values)
ax.set(xlabel='Year',ylabel='Revenue in $')
plt.title('Revenue per year')
plt.show()


# In[298]:


ax = sns.barplot(data=df,x=[2015,2016,2017],y=df.groupby('year')['customer_email'].count().values)
ax.set(xlabel='Year',ylabel='Customers')
plt.title('Total Customers per year')
plt.show()


# #### Here, we can see the pattern that 2016 was not a good year for the company as we had drop in revenue due to drop in number of customers

# In[299]:


temp_df = pd.DataFrame()


# In[300]:


temp_df['Year'] = [2015,2016,2017]
temp_df['Existing_customer_revenue'] = [0,existing_cus_revenue_prior,existing_cus_revenue]
temp_df['New_customer_revenue'] = [df[df['year']==2015]['net_revenue'].sum(),df[(df['customer_email'].isin(in_2016_not_2015))&(df['year']==2016)]['net_revenue'].sum(),df[(df['customer_email'].isin(in_2017_not_2016))&(df['year']==2017)]['net_revenue'].sum()]


# In[301]:


temp_df.set_index('Year').plot(kind='bar', stacked=True, color=['steelblue', 'red'])
plt.title('Total Revenue comparing New customers and Existing customers')
plt.ylabel('Total Revenue')
plt.xlabel('Year')
plt.show()


# **Above, for the year 2015, we would consider all customers as new customer. Here, we can see that almost same number of customers stayed in 2016 and 2017, but in 2017, more customers were added.**

# In[ ]:




