#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
import plotly.offline as py
py.init_notebook_mode(connected=True)


# In[2]:


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


# In[3]:


df = pd.read_csv('Stout.csv')


# In[4]:


df.info()


# #### Let's try to understand what each column represents

# * **emp_title**: Job title.
# 
# * **emp_length**: Number of years in the job, rounded down. If longer than 10 years, then this is represented by the value 10.
# 
# * **state**: Two-letter state code.
# 
# * **home_ownership**: The ownership status of the applicant's residence.
# 
# * **annual_income**: Annual income.
# 
# * **verified_income**: Type of verification of the applicant's income.
# 
# * **debt_to_income**: Debt-to-income ratio.
# 
# * **annual_income_joint**: If this is a joint application, then the annual income of the two parties applying.
# 
# * **verification_income_joint**: Type of verification of the joint income.
# 
# * **debt_to_income_joint**: Debt-to-income ratio for the two parties.
# 
# * **delinq_2y**: Delinquencies on lines of credit in the last 2 years.
# 
# * **months_since_last_delinq**: Months since the last delinquency.
# 
# * **earliest_credit_line**: Year of the applicant's earliest line of credit
# 
# * **inquiries_last_12m**: Inquiries into the applicant's credit during the last 12 months.
# 
# * **total_credit_lines**: Total number of credit lines in this applicant's credit history.
# 
# * **open_credit_lines**: Number of currently open lines of credit.
# 
# * **total_credit_limit**: Total available credit, e.g. if only credit cards, then the total of all the credit limits. This excludes a mortgage.
# 
# * **total_credit_utilized**: Total credit balance, excluding a mortgage.
# 
# * **num_collections_last_12m**: Number of collections in the last 12 months. This excludes medical collections.
# 
# * **num_historical_failed_to_pay**: The number of derogatory public records, which roughly means the number of times the applicant failed to pay.
# 
# * **months_since_90d_late**: Months since the last time the applicant was 90 days late on a payment.
# 
# * **current_accounts_delinq**: Number of accounts where the applicant is currently delinquent.
# 
# * **total_collection_amount_ever**: The total amount that the applicant has had against them in collections.
# 
# * **current_installment_accounts**: Number of installment accounts, which are (roughly) accounts with a fixed payment amount and period. A typical example might be a 36-month car loan.
# 
# * **accounts_opened_24m**: Number of new lines of credit opened in the last 24 months.
# 
# * **months_since_last_credit_inquiry**: Number of months since the last credit inquiry on this applicant.
# 
# * **num_satisfactory_accounts**: Number of satisfactory accounts.
# 
# * **num_accounts_120d_past_due**: Number of current accounts that are 120 days past due.
# 
# * **num_accounts_30d_past_due**: Number of current accounts that are 30 days past due.
# 
# * **num_active_debit_accounts**: Number of currently active bank cards.
# 
# * **total_debit_limit**: Total of all bank card limits.
# 
# * **num_total_cc_accounts**: Total number of credit card accounts in the applicant's history.
# 
# * **num_open_cc_accounts**: Total number of currently open credit card accounts.
# 
# * **num_cc_carrying_balance**: Number of credit cards that are carrying a balance.
# 
# * **num_mort_accounts**: Number of mortgage accounts.
# 
# * **account_never_delinq_percent**: Percent of all lines of credit where the applicant was never delinquent.
# 
# * **tax_liens**: a numeric vector
# 
# * **public_record_bankrupt**: Number of bankruptcies listed in the public record for this applicant.
# 
# * **loan_purpose**: The category for the purpose of the loan.
# 
# * **application_type**: The type of application**: either individual or joint.
# 
# * **loan_amount**: The amount of the loan the applicant received.
# 
# * **term**: The number of months of the loan the applicant received.
# 
# * **interest_rate**: Interest rate of the loan the applicant received.
# 
# * **installment**: Monthly payment for the loan the applicant received.
# 
# * **grade**: Grade associated with the loan.
# 
# * **sub_grade**: Detailed grade associated with the loan.
# 
# * **issue_month**: Month the loan was issued.
# 
# * **loan_status**: Status of the loan.
# 
# * **initial_listing_status**: Initial listing status of the loan. (I think this has to do with whether the lender provided the entire loan or if the loan is across multiple lenders.)
# 
# * **disbursement_method**: Dispersement method of the loan.
# 
# * **balance**: Current balance on the loan.
# 
# * **paid_total**: Total that has been paid on the loan by the applicant.
# 
# * **paid_principal**: The difference between the original loan amount and the current balance on the loan.
# 
# * **paid_interest**: The amount of interest paid so far by the applicant.
# 
# * **paid_late_fees**: Late fees paid by the applicant

# # Imputing the missing data 

# In[5]:


plt.figure(figsize=(25,6))
sns.heatmap(df.isna())
plt.title('Heatmap of missing values of ')


# Here, first we have to deal with the missing data. Let's try to impute the missing data.

# Now, let's take the columns name where missing values are there. 

# In[6]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Here, these 10 columns have missing values and let's figure out if we can fill these values using other given variables or not. Otherwise we might have to fill using proven imputation techniques.

# First of all, the `emp_title` can not be imputed effectively. Though we can calculate the distribution of these job titles and we can use this distribution to fill the imputed values by randomly picking the values from this distribution. But here, the problem we might face is that the job title with highest percentage would be picked always which might make our visualizations biased towards that job title. Also, there are more than 4000 unique job titles so it would be difficult to create a distribution since there are a lot of occupations that are only mentioned once. 
# 
# **Most importantly, most missing columns can be easily linked with the `emp_title`. So we can also drop the rows where `emp_title` is empty so that we don't make any wrong decision based on that.**

# Before that we first need to strip empty white spaces from left and right.

# In[7]:


df['emp_title'] = df['emp_title'].str.rstrip()


# In[8]:


df['emp_title'] = df['emp_title'].str.lstrip()


# In[9]:


# df = pd.read_csv('Stout.csv')


# In[10]:


def random_picker(row):
    if pd.isna(row['emp_title']):
        random =  df['emp_title'].value_counts(normalize=True)
        return np.random.choice(random.index, p=random.values)
    else:
        return row['emp_title']


# In[11]:


# df['emp_title'].fillna(value=str(np.random.choice(df['emp_title'].value_counts(normalize=True).index, p=df['emp_title'].value_counts(normalize=True).values)),inplace=True)
df['emp_title'] = df.apply(random_picker,axis=1)


# Now, we have removed these missing values, let's move on to other missing variables. Let's work on `emp_length` which does not have any more missing values after droppping the empty rows in `emp_title`

# ##### We have three options here: 
# * We take the probability distribution of these employment length and use that to impute missing data. Though this would not take `emp_title` into consideration and that might be somewhat inaccurate way of imputing
# 
# * (Assumption) Since positions such as Owner, Lawyer requires experience and should be aware of how company operates, we can assume that people in that role often are working in that position for more than 10 years. We can impute this missing values based on the `emp_title`. One drawback of this is that we have more than 4000 unique job titles and some job titles have only one value to compare it with. That might be inaccuracte.
# 
# * We can impute the `emp_length` with `interest_rate` or some other variable with regression imputation, etc. Though `emp_length` was not similar to any other variable so we can drop this option.

# In[12]:


df['emp_length'].value_counts()


# In[13]:


def random_picker(row):
    if pd.isna(row['emp_length']):
        random =  df[df['emp_title']==row['emp_title']]['emp_length'].value_counts(normalize=True)
        return np.random.choice(random.index, p=random.values)
    else:
        return row['emp_length']


# In[14]:


# df['emp_length'] = df['emp_length'].apply(lambda x: np.random.choice(df[df['emp_title']==df.loc[4,'emp_title']]['emp_length'].value_counts(normalize=True).index, 
#                  p=df[df['emp_title']==df.loc[4,'emp_title']]['emp_length'].value_counts(normalize=True).values) if pd.isna(x) else x)
df['emp_length'] = df.apply(random_picker,axis=1)


# In[15]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Let's analyze the `annual_income_joint`

# * We take the probability distribution of these annual joint income and use that to impute missing data. Though this would not take `emp_title` into consideration and that might be somewhat inaccurate way of imputing
# 

# **Here, in many variables, valus are Null because the application type is single so we won't have any record for that.**

# In[16]:


sns.scatterplot(data=df,x='annual_income',y='annual_income_joint')#trying third option


# **We have two option here**:
# 
# * Since, `annual_income_joint` has high correlation with the `annual_income`, let's try to find the slope and intercept and impute the missing values. Note that in practice we can take more variables in condsideration as well.
# 
# * We can also, compute the `job_title` wise mean and imptute missing values using that as well.
# 
# * Both of them have a significant drawback as the proportion of missing data is huge so if we compute using regression technique, we might not get the result that we are looking for.

# In[17]:


temp_df = df.copy()
temp_df = temp_df[temp_df['annual_income_joint'].notna()]
slope = np.corrcoef(temp_df['annual_income'],temp_df['annual_income_joint'])[0][1] * (temp_df['annual_income_joint'].std()/temp_df['annual_income'].std())
intercept = temp_df['annual_income_joint'].mean() - temp_df['annual_income'].mean()*slope
intercept + slope * 35000


# In[18]:


df.reset_index(drop=True,inplace=True)


# In[19]:


df['annual_income_joint'] = [intercept+df.loc[i,'annual_income']*slope if pd.isna(df.loc[0,'annual_income_joint']) else df.loc[i,'annual_income_joint'] for i in range(len(df))]


# In[20]:


df.head()


# In[21]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Now, let's fill the one missing value in `debt_to_income`. Since there is only one missing value, we can just use median or mean imputation as we have thousands of records and this won't affect the distribution of the column.

# In[22]:


df['debt_to_income'].fillna(value=df['debt_to_income'].mean(),inplace=True)


# In[23]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Similarly, we also impute the `debt_to_income_joint` from `debt_to_income` column using the regression analysis like we did in the `annual_income_joint` and `annual_income`.

# In[24]:


sns.scatterplot(data=df,x='debt_to_income',y='debt_to_income_joint')#trying third option
plt.xlim(0,100)#ignoring outliers from the scatterplot


# Here as well we have high correlation between these two variable. Let's fit the regression line

# In[25]:


# sns.heatmap(temp_df[['debt_to_income','debt_to_income_joint','interest_rate','balance','installment','paid_total','paid_interest']].corr(),annot=True)
plt.figure(figsize=(20,20))
sns.heatmap(temp_df.corr(),annot=True)


# There is not significant correlation found here, so let's use **Sklearn's KNNImputer** instead.

# In[26]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3, weights="uniform")
X = df[['debt_to_income','debt_to_income_joint','interest_rate','total_credit_utilized','current_installment_accounts','num_cc_carrying_balance','paid_interest','num_active_debit_accounts']]
X = imputer.fit_transform(X)


# In[27]:


df['debt_to_income_joint'] = X[:,1]


# In[28]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Now, let's deal with above five columns. First, let's analyze `verification_income_joint`.

# In[29]:


df['verification_income_joint'].value_counts()


# In[30]:


sns.stripplot(data=df,x='verification_income_joint',y='interest_rate')


# Here, we can analyze that whether user is verified or not, does not have any significant relation with interest rate. We can either drop this column or impute the missing data using the distribution of the column. 

# In[31]:


def random_picker(row):
    if pd.isna(row['verification_income_joint']):
        random =  df['verification_income_joint'].value_counts(normalize=True)
        return np.random.choice(random.index, p=random.values)
    else:
        return row['verification_income_joint']


# In[32]:


df['verification_income_joint'].value_counts()


# In[33]:


df['verification_income_joint'] = df.apply(random_picker,axis=1)


# In[34]:


# verified_or_not = df['verification_income_joint'].value_counts(normalize=True)
# df['verification_income_joint'].fillna(value=np.random.choice(verified_or_not.index, p=verified_or_not.values),inplace=True)


# In[35]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()
# empty_columns.head()


# Now let's impute `months_since_last_delinq`

# In[36]:


sns.scatterplot(data=df,x = 'months_since_last_delinq',y='interest_rate')


# This column does not seem that important. We can drop this column

# In[37]:


df.drop('months_since_last_delinq',axis=1,inplace=True)


# In[38]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()


# Now let's impute the `months_since_90d_late`

# In[39]:


sns.scatterplot(data=df,x = 'months_since_90d_late',y='interest_rate')


# Here, we can analyze that interest rate does not have any significant relation with `months_since_90d_late`. Maybe we can drop this column altogether. Also, it had highest correlation with the `months_since_last_delinq` but both had missing values at similar index so we couldn't use either to compute missing values for the other. So our safe bet is to drop this column

# In[40]:


df.drop('months_since_90d_late',axis=1,inplace=True)


# In[41]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()


# Now, let's impute `months_since_last_credit_inquiry`

# Since there are not a lot of missing data in this column, we can use KNNImputer.

# In[42]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3, weights="uniform")


# In[43]:


X = df[['debt_to_income','months_since_last_credit_inquiry','interest_rate','total_credit_utilized','current_installment_accounts','num_cc_carrying_balance','paid_interest','num_active_debit_accounts']]
X = imputer.fit_transform(X)


# In[44]:


df['months_since_last_credit_inquiry'] = X[:,1]


# In[45]:


df[df.isna().sum()[df.isna().sum()!=0].index].head()


# Now, only `num_accounts_120d_past_due` is left to impute

# In[46]:


df['num_accounts_120d_past_due'].value_counts()


# As we can see, this column does not provide any significant insight so let's just drop this column.

# In[47]:


df.drop('num_accounts_120d_past_due',axis=1,inplace=True)


# In[48]:


df['num_accounts_30d_past_due'].value_counts()


# In[49]:


df['current_accounts_delinq'].value_counts()


# #### Here, We have similar two columns as we can see above. They don't provide any valuable information so let's drop them as well. 

# In[50]:


df.drop(['num_accounts_30d_past_due','current_accounts_delinq'],axis=1,inplace=True)


# This dataset is now free of missing values.

# In[51]:


df.head(5)


# In[52]:


# df.to_csv('imputed_case_study_1.csv',index=False)


# # Preprocessing and Feature Engineering to evaluate important variables

# In[478]:


df = pd.read_csv('imputed_case_study_1.csv')


# In[479]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[480]:


X = df.loc[:, df.columns != 'interest_rate']
y = df.loc[:,'interest_rate']


# In[481]:


X.head()


# In[482]:


df['emp_title'].value_counts(normalize=True)[:10]


# First, we have to deal with the `emp_title` as it would bring high cardinality if we convert it to one-hot encoding format. When we have a variable with such high cardinality, we have three options to deal with it:
# 
# * We can lower the cardinality using featurehasher but even then we have more than 4000 unique values in `emp_title` column which is not practical. 
# 
# * We can not use this column at all for further evaluation.
# 
# * We can use first few values with highest cardinality and use them as one-hot encoded variable and out rest as 'Other' column. This would be also practical if we had more data in each category. Here, we have 0.02 of the whole data as one category which is extremely low. This would just make 'Other' variable as 1 in most cases and give us incorrect results if we use it to train our model.

# So let's not include this variable for now.

# In[483]:


X = X.loc[:,X.columns!='emp_title']


# In[484]:


sns.stripplot(data=df,x='interest_rate',y='application_type')


# Since, application type does not affect the interest rate, we can just neglect variables with joint values.

# In[485]:


onehot = pd.get_dummies(X[['homeownership','verified_income','verification_income_joint','loan_purpose','application_type',
                  'grade','sub_grade','initial_listing_status','disbursement_method','loan_status']])


# In[486]:


X.drop(X[['homeownership','verified_income','verification_income_joint','loan_purpose','application_type',
                  'grade','sub_grade','initial_listing_status','disbursement_method','loan_status']],axis=1,inplace=True)


# In[487]:


X.drop(['state'],inplace=True,axis=1)


# In[488]:


X = pd.concat([X,onehot],axis=1)


# In[489]:


X.head()


# Now let's convert the month into datetime index.

# In[490]:


X['Month'] = pd.to_datetime(df['issue_month'],format='%b-%Y').dt.month
X['year'] = pd.to_datetime(df['issue_month'],format='%b-%Y').dt.year


# In[491]:


X.drop('issue_month',axis=1,inplace=True)


# In[492]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[493]:


sel = SelectFromModel(RandomForestRegressor(n_estimators = 100),threshold=0.0005)
sel.fit(X_train, y_train)


# In[494]:


sel.get_support()


# In[495]:


selected_feat= X.columns[(sel.get_support())]


# In[496]:


selected_feat


# Here, only significant features we have for `interest_rate` is values of `Grade` and `Sub Grade`. This means that our model won't be able to perform good with these features which we have already analyzed in graphs above.

# In[497]:


from xgboost import XGBRegressor


# In[498]:


xgb = XGBRegressor()


# In[499]:


sel = SelectFromModel(xgb,threshold=0.0005)
sel.fit(X_train, y_train)


# In[500]:


sel.get_support()


# In[501]:


selected_feat= X.columns[(sel.get_support())]
selected_feat


# Similar can be seen for XGBoost as well. Let's analyze these significant features.

# # Visualization

# It is difficult to understand various grading factors which are crucial for assigning a grade to the borrower. Let's try to visualize how each factor plays a vital role in determing the interest rate.

# ## 1. Location 

# The location of a given loan is another factor to consider when making investment decisions. Each local market is different and may affect how well the completed property will sell and how much it will fetch. Lower-risk projects tend to be located in areas with strong real estate markets.

# Additionally, location also heavily influences the ease of foreclosure in the event it is necessary, which may be a factor worth considering for some investors. This is especially true in mortgage loan.

# In[585]:


fig = px.choropleth(df,
                    locations='state',
                    color='interest_rate',
                    color_continuous_scale='spectral_r',
                    hover_name='state',
                    locationmode='USA-states',
                    labels={'Interest Rate':'Interesr Rate'},
                   scope='usa')
fig.add_scattergeo(
        locations=df['state'],
    locationmode = 'USA-states',
    text=df['state'],
    mode='text'
)
fig.update_layout(title={'text':'Total Cost of Alcohol consumption by state',
                         'xanchor':'center',
                        'yanchor':'top',
                        'x':0.5})
fig.show()


# In[570]:


pivot_full_df = pd.pivot_table(df,index='state',values='interest_rate')
pivot_full_df['interest_rate'] = pivot_full_df['interest_rate'] - pivot_full_df['interest_rate'].mean()
pivot_full_df['color'] = ['red' if x < 0 else 'green' for x in pivot_full_df['interest_rate']]
pivot_full_df.sort_values('interest_rate',inplace=True)
pivot_full_df.reset_index(inplace=True)
pivot_full_df['interest_rate_real'] = df['interest_rate']
Diverging = go.Figure() 
Diverging.add_trace(go.Bar(x = pivot_full_df.interest_rate,y=pivot_full_df.state,orientation='h',width=0.5,
                           marker_color=pivot_full_df.color,customdata=pivot_full_df.interest_rate_real,
                           hovertemplate="%{y} Relative Value : %{x} Absolute Value : %{customdata}"))
Diverging.update_layout(barmode='relative',height=1000,width=900,legend_orientation='v',title={'text':'Relative interest rate of loans in US States'})
Diverging.update_xaxes(title='Relative Interest Rate')
Diverging.update_yaxes(title='U.S. States')


# **Surprisingly, the state with highest overall interest rate is Hawaii which actually has the lowest mortgage interest rate.** Here, we are considering personal loan as well so that can be the reason why Hawaii has higher interest rate. Though we need to filter out these records based on type of loan, etc to gain more insight.

# **We have to refer state-specific laws and interest rate set by the Federal reserve to keep track of the interest rate accurately.**

# Here, we can depict which state has highest interest rate. Though interest rate depends on various factors, main creteria can be per capita income, poverty, taxes, tax systems of that state. 

# ## 2. Grade Assigned

# Loans can be assigned one of seven letter grades from A to G, and each grade generally reflects the overall risk of the loan. For example, Grade A loans generally have lower expected returns, lower expected loan losses, and corresponding lower interest payments; whereas on the other end of the spectrum, Grade G loans have higher expected returns, higher potential loan losses, but correspondingly higher interest rates. With Groundfloor, you create a custom portfolio of real estate investments based on your own investment criteria and risk tolerances. 

# In[510]:


fig = px.strip(df, x="sub_grade", y="interest_rate",category_orders={'sub_grade':['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                                                                             'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                                                                             'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5',
                                                                             'G1','G4']},width=1000, height=600,
                                                                              title='Sub-grade wise distribution of interest rates')
fig.add_vrect(x0='-0.5',x1='4.5',annotation_text='Grade A',fillcolor='Green',opacity=0.1,annotation_position='inside left')
fig.add_vrect(x0='4.5',x1='9.5',annotation_text='Grade B',fillcolor='Blue',opacity=0.1,annotation_position='inside left')
fig.add_vrect(x0='9.5',x1='14.5',annotation_text='Grade C',fillcolor='Green',opacity=0.1,annotation_position='outside right')
fig.add_vrect(x0='14.5',x1='19.5',annotation_text='Grade D',fillcolor='Blue',opacity=0.1,annotation_position='top left')
fig.add_vrect(x0='19.5',x1='24.5',annotation_text='Grade E',fillcolor='Green',opacity=0.1,annotation_position='inside left')
fig.add_vrect(x0='24.5',x1='29.5',annotation_text='Grade F',fillcolor='Blue',opacity=0.1,annotation_position='inside left')
fig.add_vrect(x0='29.5',x1='31.5',annotation_text='Grade G',fillcolor='Green',opacity=0.1,annotation_position='inside left')
fig.update_xaxes(title='Sub grades')
fig.update_yaxes(title='Interest Rates')
fig.show()


# Above we can depict that interest starts rising with the drop in associated grade with that loan. This relation in the graph seems accurate. 
# 
# **As interest rate is mostly just dependent on grade of the loan, we can start analying the relation between assigned grades and various variabls** 
# 
# Moreover, we also noticed that interest rate does not rely on any other varibles given in the dataset. In practice, the grade is assigned between A1 and G5 which means that our dataset does follow that format which is a good thing. Next, the grade is determined based on FICO score which we are not given. But assuming that Grades were calculated from FICO score itself, we can take this into consideration. **credit score can be estimated with credit limit and credit used which gives us credit utilization score as portion of your FICO score is determined by credit utilization**.
# 
# 

# ## 3. Credit Utilization 

# We can plot the visualization of credit utilization for further analysis. Credit utilization rate has proven to be extremely predictive of future repayment risk. So it is often an important factor in a person's score. Generally speaking, the higher your utilization rate is, the greater is the risk that you will default on a credit account within the next two years.

# In[571]:


df['credit_utilization'] = df['total_credit_utilized']/df['total_credit_limit']
temp_df = df.groupby(['sub_grade'])['credit_utilization'].mean()
t = pd.DataFrame()
t['sub_grade'] = temp_df.index
t['credit_utilization'] = temp_df.values
t['grade'] = ['A']*5 + ['B']*5 + ['C']*5 + ['D']*5 + ['E']*5+['F']*5+['G']*2
fig = px.bar(t,x='sub_grade',y='credit_utilization',color='grade',title='Average Credit Utlization and Grades Assigned to them')
fig.update_xaxes(title='Sub Grades')
fig.update_yaxes(title='Average Credit Utilization')
fig.show()


# Here, we can depict that the credit utilization plays a vital role in determing the FICO score which is used to assign the grades to the borrower. As the credit utilization score starts rising, the grade assigned is much lower. By analyzing the above chart, **It is safe to say that 0.3 credit utilization is holds true here on this dataset as well.** Credit utilization above 0.3 can bring the grade down. 
# 
# One outlier here is grade **G4** which has low credit utilization. It is because there is only one record of having **G4** sub grade and that is why we can count is an an outlier.

# ## 4. Loan Amount

# In[586]:


fig = px.box(df,x='grade',y='loan_amount',category_orders={'grade':['A','B','C','D','E','F','G']},color='grade',title='Average Loan Amount and Grades Assigned to them')
fig.update_xaxes(title='Grades')
fig.update_yaxes(title='Average Loan Amount')
fig.show()
fig.write_html("images.html")


# Here, the loan amount also brings down your grade as it depends on your credit limit, etc and when we ask to borrow more money compared to our credit limit, the grade can go down. Important thing to note here is that the distribution is quite dispersed for each Grade. But for grade G, the total Loan amount is quite high which suggests that because the users had bad credit score and the total loan amount is higher, the chances of them having a lower grade is obvious.

# # Training the model 

# In[650]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


# In[615]:


df = pd.read_csv('imputed_case_study_1.csv')


# In[616]:


X = df.loc[:, df.columns != 'interest_rate']
y = df.loc[:,'interest_rate']


# In[617]:


X.head()


# First, we have to deal with the `emp_title` as it would bring high cardinality if we convert it to one-hot encoding format. When we have a variable with such high cardinality, we have three options to deal with it:
# 
# * We can lower the cardinality using featurehasher but even then we have more than 4000 unique values in `emp_title` column which is not practical. 
# 
# * We can not use this column at all for further evaluation.
# 
# * We can use first few values with highest cardinality and use them as one-hot encoded variable and out rest as 'Other' column. This would be also practical if we had more data in each category. Here, we have 0.02 of the whole data as one category which is extremely low. This would just make 'Other' variable as 1 in most cases and give us incorrect results if we use it to train our model.

# So let's not include this variable for now.

# In[618]:


X = X.loc[:,X.columns!='emp_title']


# Since, application type does not affect the interest rate, we can just neglect variables with joint values.

# In[619]:


onehot = pd.get_dummies(X[['homeownership','verified_income','verification_income_joint','loan_purpose','application_type',
                  'grade','sub_grade','initial_listing_status','disbursement_method','loan_status']])


# In[620]:


X.drop(X[['homeownership','verified_income','verification_income_joint','loan_purpose','application_type',
                  'grade','sub_grade','initial_listing_status','disbursement_method','loan_status']],axis=1,inplace=True)


# In[621]:


X.drop(['state'],inplace=True,axis=1)


# In[622]:


X = pd.concat([X,onehot],axis=1)


# In[623]:


X.head()


# Now let's convert the month into datetime index.

# In[624]:


X['Month'] = pd.to_datetime(df['issue_month'],format='%b-%Y').dt.month
X['year'] = pd.to_datetime(df['issue_month'],format='%b-%Y').dt.year


# In[625]:


X.drop('issue_month',axis=1,inplace=True)


# In[626]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[628]:


sel = SelectFromModel(RandomForestRegressor(n_estimators = 100),threshold=0.001)
sel.fit(X_train, y_train)


# In[629]:


sel.get_support()


# In[630]:


selected_feat= X.columns[(sel.get_support())]


# In[641]:


train_X = X_train[selected_feat]


# In[644]:


test_X = X_test[selected_feat]


# ### Since all the columns that are relevant are one-hot encoded variables, we don't have to Standardize or Normalize our data. Also, since we are working witht the one-hot encoded variables, Random Forest or XGBoost would be a better option. 

# In[652]:


import optuna
import sklearn
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10,1000)
    max_depth = int(trial.suggest_int('max_depth', 1,1000))
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 100)
    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,max_features=max_features,
                                                  max_leaf_nodes=max_leaf_nodes)
    clf.fit(train_X,y_train)
    result = clf.predict(test_X)
    return mean_squared_error(y_test,result)


# In[653]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)


# In[654]:


trial = study.best_trial
print('Mean Squared Error: {}'.format(trial.value))


# In[655]:


print("Best hyperparameters: {}".format(trial.params))


# In[657]:


optimised_rf = RandomForestRegressor(max_depth = study.best_params['max_depth'], 
                                     max_features = study.best_params['max_features'],
                                     max_leaf_nodes = study.best_params['max_leaf_nodes'],
                                     n_estimators = study.best_params['n_estimators'],
                                     n_jobs=-1)


# In[658]:


optimised_rf.fit(train_X,y_train)


# In[659]:


result = optimised_rf.predict(test_X)


# In[661]:


print('Mean Squared Error here is: ',mean_squared_error(y_test,result))


# **Here, this concludes the model training using `RandomForestRegressor`**

# In[664]:


from xgboost import XGBRegressor


# In[665]:


xgb = XGBRegressor()


# In[666]:


sel = SelectFromModel(xgb,threshold=0.001)
sel.fit(X_train, y_train)


# In[667]:


sel.get_support()


# In[668]:


selected_feat= X.columns[(sel.get_support())]
selected_feat


# In[669]:


train_X = X_train[selected_feat]
test_X = X_test[selected_feat]


# In[676]:


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10,1000)
    max_depth = int(trial.suggest_int('max_depth', 1,1000))
    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
    xgb.fit(train_X,y_train)
    result = xgb.predict(test_X)
    return mean_squared_error(y_test,result)


# In[677]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)


# In[678]:


trial = study.best_trial
print('Mean Squared Error: {}'.format(trial.value))


# In[679]:


print("Best hyperparameters: {}".format(trial.params))


# In[680]:


optimised_xg = XGBRegressor(max_depth = study.best_params['max_depth'], 
                                     n_estimators = study.best_params['n_estimators'],
                                     n_jobs=-1)


# In[681]:


optimised_xg.fit(train_X,y_train)


# In[682]:


result = optimised_xg.predict(test_X)


# In[683]:


print('Mean Squared Error here is: ',mean_squared_error(y_test,result))


# **Here, the MSE is quite good which is extremelyy good score considering the related features are all one-hot encoded variables.**

# Here, we can tune some more hyperparameters but as focus of this take-home was visualizations, we can do that later.

# 
# 
# # Issues with the dataset 

# * The major issue with the dataset is that it doesn't make any sense if you try to analyze the relations between each variable. Clearly it is a dummy dataset or it was refined a lot. Many relations between two varibles are uniformly distributed.
# 
# * The `emp_title` does some typos and the `emp_title` could have been segregated into fewer roles as well. We have more than **4000** unique roles which are so difficult to analyze all at once and make valid visualizations. For example, we have values such as "6th grade teacher", "teacher",etc which we don't require unless some special case.

# # Assumptions 

# * Most missing columns can be easily linked with the `emp_title`. So we can drop the rows where `emp_title` is empty so that we don't make any wrong decision based on that. 
# * Since positions such as Manager, Owner requires experience and should be aware of how company operates, we can assume that people in that role often are working in that position for more than 10 years. We can impute `emp_length` missing values based on the `emp_title`.

# # Future Work 

# * Impute missing values by calculating correlation comparing continuos variable and categorical variables using **Point-Biserial Correlation**.
# * Find the relation between each variable and grade/interest rate to get more insight.
# * Try more models for prediction

# In[ ]:




