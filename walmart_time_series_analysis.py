#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Libraries

# In[1]:


import numpy as np          #To perform mathematical calculation or array operation
import pandas as pd         # To use DataFrame
from pandas.plotting import autocorrelation_plot as auto_corr

#To plot
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# For date-time
import math
from datetime import datetime,timedelta

# Other libraries
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose as season
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.arima_model import ARIMA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics


import warnings
warnings.filterwarnings("ignore")


# In[2]:


get_ipython().system('pip install pmdarima')
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


# In[3]:


pd.options.display.max_columns=100  #To see all columns

sns.set_style("darkgrid")

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10,6)
matplotlib.rcParams["figure.facecolor"] = "#00000000"


# In[4]:


df_train = pd.read_csv("Walmart train.csv")          #train set
df_store = pd.read_csv("Walmart stores .csv")        # Store data
df_features = pd.read_csv("Walmart features.csv")    #External information


# First of all look at data then merge three dataframe

# In[5]:


df_train.head()


# In[6]:


df_store.head()


# In[7]:


df_features.head()


# In[8]:


#merging 3 different sets
df = df_train.merge(df_features,on=["Store","Date"],how="inner").merge(df_store,on=["Store"],how="inner")
df.head()


# In[9]:


df.drop(["IsHoliday_y"],axis=1,inplace=True)  #Removing duplicate column


# In[10]:


df.rename(columns={"IsHoliday_x":"IsHoliday"},inplace=True)  #Rename the column


# In[11]:


df.head()   #last ready dataset


# In[12]:


print("Total Number of Rows is: ",df.shape[0])
print("Total Number of Columns is: ",df.shape[1])


# In[13]:


df.info()       #Check information about the dataset


# - markdown columns contain null values.
# - It usas total 52 MB memory.
# - It has only 2 categorical feature. Lets see the number of unique value in these feature.

# **Store And Department**

# In[14]:


print("Store has {} unique values.".format(df["Store"].nunique()))
print("Dept has {} unique values.".format(df["Dept"].nunique()))


# Check the average sales for each store and department.

# In[15]:


store_dept_table = pd.pivot_table(df,index="Store",columns="Dept",values="Weekly_Sales",aggfunc=np.mean)
store_dept_table


# - There are 45 stores available.
# - Department numbers are begin from 1 to 99 but some deparment number is missing such as 88 and 89 etc. 
# - We can see the wrong values in some store and department like negative value and zero. But sales can not be minus. And it is impossible for one department not to see anything whole week. So let's change these value.

# In[16]:


df.loc[df["Weekly_Sales"]<=0]


# 1358 rows in 421570 rows means 0.3%,so I can delete and ignore these rows which contains wrong sales values.

# In[17]:


df = df.loc[df["Weekly_Sales"] > 0]
df.shape


# **IsHoliday**

# In[18]:


df["IsHoliday"].unique()


# In[19]:


sns.barplot(x="IsHoliday",y="Weekly_Sales",data=df);


# From above chart we can say that average weekly salesfor holidays are significantly higher that in holiday period. Let's see number of holiday weeks.

# In[20]:


df_holiday = df.loc[df["IsHoliday"] == True]
df_non_holiday =df.loc[df["IsHoliday"] == False]
print("Number of holiday weeks is: ",df_holiday["Date"].nunique())
print("Number of Non holiday weeks is: ",df_non_holiday["Date"].nunique())


# In train data, there are 133 weeks for non-holiday and 10 weeks for holiday.

# In[21]:


# Check the holidays date
df_holiday["Date"].unique()


# from the description of dataset, we know that there are 4 holidays values such as:
# 
# Super Bown: 12-Feb-10,11-Feb-11,10-Feb-12,8-Feb-13
# 
# Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
# 
# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
# 
# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
# 
# let's see differences between holiday types:

# In[22]:


#  Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False


# In[23]:


# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False


# In[24]:


# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False


# In[25]:


#Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False


# In[26]:


holiday_list = ["Christmas","Thanksgiving","Super_Bowl","Labor_Day"]

plt.figure(figsize= (15,10))
for i,feature in enumerate(holiday_list):
    plt.subplot(2,2,i+1)
    sns.barplot(x=df[feature],y=df["Weekly_Sales"])
    plt.title(feature,fontsize=15)


# It shown that for the graph, Labour day and christmas do not increase weekly average sales. There is positive effect on sales in Super bowl, but the highest difference is inthe ThanksGiving. I think people generally prefer to buy cristmas gifts 1-2 weeks before christmas, so it does not change sales in the cristmas week.

# **Type**

# In[27]:


df["Type"].unique()


# There are three different store type in the data as A,B, and C.

# In[28]:


df["Type"].value_counts()


# Type A is the highest number of store. I want to see percentages of store types.

# In[29]:


type_percentage = (df["Type"].value_counts())*100/(df["Type"].value_counts().sum())

plt.figure(figsize=(5,5))
plt.pie(type_percentage.values,labels=type_percentage.index,autopct="%1.1f%%",textprops={"fontsize":10})
plt.show()


# Half of the stores are belongs to Type A.

# In[30]:


df.groupby(["Type"])["Weekly_Sales"].mean()


# In[31]:


#Avg weekly sales for types on Christmas
sns.barplot(x="Type",y="Weekly_Sales",hue="Christmas",data=df);


# - Weekly sales does not effect by christmas.

# In[32]:


#Avg weekly sales for types on Labor_Day
sns.barplot(x="Type",y="Weekly_Sales",hue="Labor_Day",data=df);


# Weekly sales are same in holiday and non_holidays period for Type A and Type B but in Type C there is positive change in sales for holiday period.

# In[33]:


#Avg weekly sales for types on Thanksgiving
sns.barplot(x="Type",y="Weekly_Sales",hue="Thanksgiving",data=df);


# Weekly sales shows high positive growth in Type A and Type B for holiday period.

# In[34]:


#Avg weekly sales for types on Super_Bowl
sns.barplot(x="Type",y="Weekly_Sales",hue="Super_Bowl",data=df);


# Weekly sales shows postive changes in every type of store for holiday period.

# **Size**
# 
# As size is an integer type, let's see size and type relation

# In[35]:


df_store.groupby("Type").describe()["Size"].round(2)


# In[36]:


# To see the type-size relation
fig = sns.boxplot(x="Type",y="Size",data=df,showfliers=False)


# Size of the type of stores are consistent with sales, as expected. Higher size stores has higher sales.

# **Markdown Columns**
# 
# Walmart gave markdown columns to see the effect if markdowns on sales. When I check columns, there are many NaN values for markdowns. I decided to change them with 0, because if there is markdown in the row, it is shown with numbres. So, if I can write 0, it shows there is no markdown at that date.

# In[37]:


df.isna().sum()


# In[38]:


df = df.fillna(0)  #filling null's with 0
df.isna().sum()


# In[39]:


df.describe() # to see  statistical things


# Minimum value for weekly sales is 0.01. Most probably, this value is not true but I prefer not to change them now. Because, there are many departments and many stores. It takes too much time to check each department for each store (45 store for 81 departments). So, I take averages for EDA.

# **Sales and Dept**

# In[40]:


x = df["Dept"]
y = df["Weekly_Sales"]
plt.scatter(x,y)
plt.xlabel('Departments')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales by Department')
plt.show()


#  It is seen that one department between 60-80(I assume it is 72), has higher sales values.

# In[41]:


plt.figure(figsize=(30,10))
sns.barplot(x='Dept', y='Weekly_Sales', data=df);


# it is seen that department 92 has higher mean sales.

# **Sales and Store**

# In[42]:


x = df["Store"]
y = df["Weekly_Sales"]
plt.scatter(x,y)
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales by Store')
plt.show()


# In[43]:


plt.figure(figsize=(20,10))
sns.barplot(x='Store', y='Weekly_Sales', data=df);


#  From the first graph, some stores has higher sales but on average store 20 is the best and 4 and 14 following it.
#  
#  **Datetime**

# In[44]:


df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
df['week'] =df['Date'].dt.week
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year


# In[45]:


# to see the best months for sales
df.groupby('month')['Weekly_Sales'].mean()


# In[46]:


## to see the best years for sales
df.groupby('year')['Weekly_Sales'].mean()


# In[47]:


#Let's visualize mothly sales and yearly sales
monthly_sales = pd.pivot_table(df,values="Weekly_Sales",columns="year",index="month")
monthly_sales.plot();


# From the graph, it is seen that 2011 has lower sales than 2010 generally.2012 has no information about November and December which have higher sales.Despite of 2012 has no last two months sales.

# In[48]:


sns.barplot(x='month', y='Weekly_Sales', data=df);


# When we look at the graph above, the best sales are in December and November, as expected. The highest values are belongs to Thankgiving holiday but when we take average it is obvious that December has the best value.

# In[49]:


df.groupby('week')['Weekly_Sales'].mean().sort_values(ascending=False).head()


# Top 5 sales averages by weekly belongs to 1-2 weeks before Christmas, Thanksgiving, Black Friday and end of May, when the schools are closed.

# In[50]:


weekly_sales = pd.pivot_table(df,values="Weekly_Sales",columns="year",index="week")
weekly_sales.plot();


# In[51]:


plt.figure(figsize=(20,6))
sns.barplot(x='week', y='Weekly_Sales', data=df);


# From graphs, it is seen that 51th week and 47th weeks have significantly higher averages as Christmas, Thankgiving and Black Friday effects.

# **Fuel Price,CPI,Unemployment,Temprerature**

# In[52]:


fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot();


# In[53]:


temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
temp.plot();


# In[54]:


CPI = pd.pivot_table(df, values = "Weekly_Sales", index= "CPI")
CPI.plot();


# In[55]:


unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot();


# From graphs, it is seen that there are no significant patterns between CPI, temperature, unemployment rate, fuel price vs weekly sales. There is no data for CPI between 140-180 also.

# ## Finding and Explorations
# 
# ### Cleaning Process
# 
# - The data has no too much missing values. All columns was checked.
# - I choose rows which has higher than 0 weekly sales.Minus values are 0.3% of data. So, I dropped them.
# - Null values in markdowns changed to zero. Because, they were written as null of there were no markdown on this department.
# 
# ### Explorations & Findings
# 
# - There are 45 stores and 81 department in data. Departments are not same in allstores.
# - Although department 72 has higher weekly saleds values,on average department 92 is the best.It shows ud, some departments has higher values as seasonal like Thanksgiving.It is consistant when we look at the top 5 sales in data, all of them belongs to 72th department at Thanksgiving holiday time.
# - Although stores 10 and 35 have higher weekly sales values sometimes, in general average store 20 and store 4 are on the first and second rank. It means that some areas has higher seasonal sales.
# - Stores has 3 types as A, B and C according to their sizes. Almost half of the stores are bigger than 150000 and categorized as A. According to type, sales of the stores are changing.
# - As expected, holiday average sales are higher than normal dates.
# - Christmas holiday introduces as the last days of the year. But people generally shop at 51th week. So, when we look at the total sales of holidays, Thankgiving has higher sales between them which was assigned by Walmart.
# - Year 2010 has higher sales than 2011 and 2012. But, November and December sales are not in the data for 2012. Even without highest sale months, 2012 is not significantly less than 2010, so after adding last two months, it can be first.
# - It is obviously seen that week 51 and 47 have higher values and 50-48 weeks follow them. Interestingly, 5th top sales belongs to 22th week of the year. This results show that Christmas, Thankgiving and Black Friday are very important than other weeks for sales and 5th important time is 22th week of the year and it is end of the May, when schools are closed. Most probably, people are preparing for holiday at the end of the May.
# - January sales are significantly less than other months. This is the result of November and December high sales. After two high sales month, people prefer to pay less on January.
# - CPI, temperature, unemployment rate and fuel price have no pattern on weekly sales.

# In[56]:


df


# ### Encoding the Data

# For preprocessing our data, I will change holidays boolean values to 0-1 and replace type of the store from A,B,C to 1,2,3.

# In[57]:


df_encoded = df.copy()  #To keep original deta frame taking copy of it


# In[58]:


type_group = {"A":1,"B":2,"C":3}  #Changing A,B,C to 1,2,3
df_encoded["Type"] = df_encoded["Type"].replace(type_group)


# In[59]:


df_encoded["Super_Bowl"] = df_encoded["Super_Bowl"].astype(bool).astype(int) #Changing True, False to 0,1
df_encoded['Thanksgiving'] = df_encoded['Thanksgiving'].astype(bool).astype(int) # #Changing True, False to 0,1
df_encoded['Labor_Day'] = df_encoded['Labor_Day'].astype(bool).astype(int) # #Changing True, False to 0,1
df_encoded['Christmas'] = df_encoded['Christmas'].astype(bool).astype(int) # #Changing True, False to 0,1
df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(bool).astype(int) # #Changing True, False to 0,1


# In[60]:


df_new = df_encoded.copy() # taking the copy of encoded df to keep it original


# ### Observation of Interactions between Features

# In[61]:


drop_col = ['Super_Bowl','Labor_Day','Thanksgiving','Christmas']
df_new.drop(drop_col, axis=1, inplace=True) # dropping columns


# In[62]:


sns.heatmap(df_new.corr().abs())    # To see the correlations
plt.show()


# Temperature, unemployment, CPI have no significant effect on weekly sales, so I will drop them. Also, Markdown 4 and 5 highly correlated with Markdown 1. So, I will drop them also. It can create multicollinearity problem, maybe. So, first I will try without them.

# In[63]:


drop_col = ['Temperature','MarkDown4','MarkDown5','CPI','Unemployment']
df_new.drop(drop_col, axis=1, inplace=True) # dropping columns


# In[64]:


sns.heatmap(df_new.corr().abs())    # To see the correlations without dropping columns
plt.show()


# Size and type are highly correlated with weekly sales. Also, department and store are correlated with sales.

# In[65]:


df_new = df_new.sort_values(by='Date', ascending=True) # sorting according to date


# ### Creating Train-Test Splits
# 
# Our data columns has continuos values, to keep the date features continue, I will not random spliting. So, I split data manually according to 70%.

# In[66]:


train_data = df_new[:int(0.7*(len(df_new)))] #taking train part
test_data = df_new[int(0.7*(len(df_new))):]  # taking test part

target = "Weekly_Sales"
used_cols = [c for c in df_new.columns.to_list() if c not in [target]] #all columns except weekly sales

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]


# We have enough information in our date such as week of the year. So, I drop date columns.

# In[67]:


X_train = X_train.drop(['Date'], axis=1) # dropping date from train
X_test = X_test.drop(['Date'], axis=1) # dropping date from test


# ### Metric Definition Function
# 
# Our metric is not calculated as default from ready models. It is weighed error so, I will use function below to calculate it.

# In[68]:


def wmae_test(test, pred): # WMAE for test 
    weights = X_test['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error


# ### Random Forest Regressor
# 
# To tune the regressor, I can use gridsearch but it takes too much time for this type of data which has many rows abd columns. So I choose regressor parameters manually.I changed the parameters each time and try to find the beast result.

# In[69]:


rf = RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1,max_depth=35,max_features="sqrt",
                          min_samples_split=10)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

#Making pipe to use scaler and regressor together
pipe = make_pipeline(scaler,rf)
pipe.fit(X_train,y_train)

# Predictions on train set
y_pred = pipe.predict(X_train)

# Prediction on test set
y_pred_test = pipe.predict(X_test)


# In[70]:


wmae_test(y_test, y_pred_test)


# For the first trial, my weighted error is around 5700..

# **To See Feature Importance**

# In[71]:


importance_df = pd.DataFrame({
    "feature":X_train.columns,
    "importance":rf.feature_importances_
}).sort_values("importance",ascending=False)
importance_df


# In[72]:


plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='feature', y='importance')
plt.xticks(rotation=75);


# After looking feature importance,I dropped month column because month column is highly correlated with week and tried the model.

# In[73]:


X1_train = X_train.drop(["month"],axis=1)   #Droping month
X1_test = X_test.drop(["month"],axis=1)


# In[74]:


rf = RandomForestRegressor(n_estimators=50,random_state=42,max_depth=35,max_features="sqrt",min_samples_split=10)
scaler = RobustScaler()
pipe = make_pipeline(scaler,rf)
pipe.fit(X1_train,y_train)

#  Predictions on train set
y_pred = pipe.predict(X1_train)

# Predictions on test set
y_pred_test = pipe.predict(X1_test)


# In[75]:


wmae_test(y_test, y_pred_test)


# It gives better results than baseline.

# **Model with Whole Data**
# 
# Now, I want to make sure that my model will learn from the columns which I dropped or not. So, I will apply my model to whole encoded data again.

# In[76]:


#Spliting train-test to whole dataset
train_data_enc = df_encoded[:int(0.7*(len(df_encoded)))]
test_data_enc = df_encoded[int(0.7*(len(df_encoded))):]

target = "Weekly_Sales"
used_cols1 = [c for c in df_encoded.columns.to_list() if c not in [target]] #all columns except price

X_train_enc = train_data_enc[used_cols1]
X_test_enc = test_data_enc[used_cols1]
y_train_enc = train_data_enc[target]
y_test_enc = test_data_enc[target]


# In[77]:


X_enc = df_encoded[used_cols1]  #to get together train and test split


# In[78]:


X_enc = X_enc.drop(["Date"],axis=1) #Droping date column for whole X


# In[79]:


X_train_enc = X_train_enc.drop(["Date"],axis=1)  #Droping data from train and test
X_test_enc = X_test_enc.drop(["Date"],axis=1)


# In[80]:


rf = RandomForestRegressor(n_estimators=50,random_state=42,
                          n_jobs=-1,max_depth=35,max_features="sqrt",
                          min_samples_split=10)
scaler = RobustScaler()
pipe = make_pipeline(scaler,rf)
pipe.fit(X_train_enc,y_train_enc)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc)


# In[81]:


wmae_test(y_test_enc, y_pred_test_enc)


# I found better resultsfor whole data,it means ourmodel can learn from columns which i droped before.

# **To See Feature Importance**

# In[82]:


importance_df = pd.DataFrame({
    "feature":X_train_enc.columns,
    "importance":rf.feature_importances_
}).sort_values("importance",ascending=False)
importance_df


# In[83]:


plt.title('Feature Importance')
sns.barplot(data=importance_df, x='feature', y='importance')
plt.xticks(rotation=75);


# Accroding to feature importance,I dropped some columns from whole set and try my model again.

# In[84]:


df_encoded_new  =df_encoded.copy() # taking copy of encoded data to keep it without change
df_encoded_new.drop(drop_col,axis=1,inplace=True)


# **Model Accroding to Feature Importance**

# In[85]:


#train-test splitting
train_data_enc_new = df_encoded_new[:int(0.7*(len(df_encoded_new)))]
test_data_enc_new = df_encoded_new[int(0.7*(len(df_encoded_new))):]

target = "Weekly_Sales"
used_cols2 = [c for c in df_encoded_new.columns.to_list() if c not in [target]] # all columns except price

X_train_enc1 = train_data_enc_new[used_cols2]
X_test_enc1 = test_data_enc_new[used_cols2]
y_train_enc1 = train_data_enc_new[target]
y_test_enc1 = test_data_enc_new[target]

#droping date from train-test
X_train_enc1 = X_train_enc1.drop(['Date'], axis=1)
X_test_enc1= X_test_enc1.drop(['Date'], axis=1)


# In[86]:


rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=40,
                           max_features = 'log2',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train_enc1, y_train_enc1)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc1)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc1)


# In[87]:


pipe.score(X_test_enc1,y_test_enc1)


# In[88]:


wmae_test(y_test_enc1, y_pred_test_enc)


# I found best result with doing feature selection from whole encoded dataset.

# **Model with Dropping Month Column**
# 
# With the same dataset before, I try to model again without month column.

# In[89]:


df_encoded_new1 = df_encoded.copy()
df_encoded_new1.drop(drop_col,axis=1,inplace=True)


# In[90]:


df_encoded_new1 = df_encoded_new1.drop(['Date'], axis=1)
df_encoded_new1 = df_encoded_new1.drop(['month'], axis=1)


# In[91]:


#train-test split
train_data_enc_new1 = df_encoded_new1[:int(0.7*(len(df_encoded_new1)))]
test_data_enc_new1 = df_encoded_new1[int(0.7*(len(df_encoded_new1))):]

target = "Weekly_Sales"
used_cols3 = [c for c in df_encoded_new1.columns.to_list() if c not in [target]] # all columns except price

X_train_enc2 = train_data_enc_new1[used_cols3]
X_test_enc2 = test_data_enc_new1[used_cols3]
y_train_enc2 = train_data_enc_new1[target]
y_test_enc2 = test_data_enc_new1[target]


# In[92]:


#modeling part
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train_enc2, y_train_enc2)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc2)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc2)


# In[93]:


pipe.score(X_test_enc2,y_test_enc2)


# In[94]:


wmae_test(y_test_enc2, y_pred_test_enc)


# It did not give better results than before.

# In[95]:


df_results = pd.DataFrame(columns=["Model", "Info",'WMAE']) # result df for showing results together


# In[96]:


#writing results to df
df_results = df_results.append({
    "Model":"RandomForestRegressor",
    "Info": "w/out divided holiday columns",
    "WMAE":5850
},ignore_index=True)


# In[97]:


df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'w/out month column' , 
       'WMAE' : 5494}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data' , 
       'WMAE' : 2450}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data with feature selection' , 
       'WMAE' : 1801}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data with feature selection w/out month' , 
       'WMAE' : 2093}, ignore_index=True)


# In[98]:


df_results


# The best results belongs to whole data set with feature selection. Now, I will try time series models.

# ## Time Series Models

# In[99]:


df.head() #To see data


# In[100]:


df["Date"] = pd.to_datetime(df["Date"]) #Changing data to dataframe for decomposing


# In[101]:


df.set_index("Date",inplace=True) #Seting data as index


# **Plotting Sales**

# In[102]:


plt.figure(figsize=(16,6))
df["Weekly_Sales"].plot()
plt.show()


# In this data, there are lots of same data values. So I will collect them together as weekly.

# In[103]:


df_week =df.resample("W").mean()  #Resample data as weekly


# In[104]:


plt.figure(figsize=(16,6))
df_week["Weekly_Sales"].plot()
plt.title("Average Sales - Weekly")
plt.show()


# With the collecting data as weekly, I can see averge sales clearly. To see monthly pattern, I resmapled my data to monthly also.

# In[105]:


df_month = df.resample("MS").mean() #Resampling as monthly


# In[106]:


plt.figure(figsize=(16,6))
df_month["Weekly_Sales"].plot()
plt.title("Average Sales - Monthly")
plt.show()


# When I turned data to monthly, I realized that I lost some patterns in weekly data. So, I will continue with weekly resmapled data.

# **To Observe 2- weeks Rolling Mean and Std**

# In[107]:


#Finding 2-weeks rolling mean and std
roll_mean = df_week["Weekly_Sales"].rolling(window=2,center=False).mean()
roll_std = df_week["Weekly_Sales"].rolling(window=2,center=False).std()


# In[108]:


fig,ax =plt.subplots(figsize=(13,6))
ax.plot(df_week["Weekly_Sales"],color="blue",label="Average Weekly Sales")
ax.plot(roll_mean,color="red",label="Rolling 2-Week Mean")
ax.plot(roll_std,color="black",label="Rolling 2-Week Std")
plt.legend()
fig.tight_layout()


# In[109]:


adfuller(df_week["Weekly_Sales"])


# From test and my observation my data is not stationary.So, I will try to find more stationary version for it.

# **Train - Test Split of Weekly Data**
# 
# To take train-test splits countinuosly, I split them manually, not random.

# In[110]:


train_data =df_week[:int(0.7*(len(df_week)))]
test_data = df_week[int(0.7*(len(df_week))):]

print("Train: ",train_data.shape)
print("Test: ",test_data.shape)


# In[111]:


target = "Weekly_Sales"
used_cols = [c for c in df_week.columns.to_list() if c not in [target]] #All columns except sales

#Assogning train-test X-y values
X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]


# In[112]:


train_data["Weekly_Sales"].plot(figsize=(20,8),title="Weekly_Sales",fontsize=14)
test_data["Weekly_Sales"].plot(figsize=(20,8),title="Weekly_Sales",fontsize=14)
plt.show()


# Blue line represents my train data, yellow is test data.

# **Decomposing Weekly Data to observe Seasonality**

# In[113]:


decomposed = decompose(df_week['Weekly_Sales'].values, 'additive', m=20) #decomposing of weekly data 


# In[114]:


decomposed_plot(decomposed, figure_kwargs={'figsize': (16, 10)})
plt.show()


# From the graphs above, every 20 steps seasonality converges to beginning point. This helps me to me tune my model.

# **Tying To Make Data More Stationary**

# Now, I will try to make my data more stationary. To do this, I will try model with differenced,logged and shifted data.

# ### 1. Difference

# In[115]:


df_week_diff = df_week["Weekly_Sales"].diff().dropna()  #Creating difference values


# In[116]:


#taking mean and std of difefrenced data
diff_roll_mean = df_week_diff.rolling(window=2,center=False).mean()
diff_roll_std = df_week_diff.rolling(window=2,center=False).std()


# In[117]:


fig,ax = plt.subplots(figsize=(13,6))
ax.plot(df_week_diff,color="blue",label="Difference")
ax.plot(diff_roll_mean,color="red",label="Rolling Mean")
ax.plot(diff_roll_std,color="black",label="Rolling Standard Deviation")
ax.legend()
fig.tight_layout()


# ### 2. Shift

# In[118]:


df_week_lag = df_week["Weekly_Sales"].shift().dropna() #Shifting the data


# In[119]:


lag_roll_mean =df_week_lag.rolling(window=2,center=False).mean()
lag_roll_std = df_week_lag.rolling(window=2,center=False).std()


# In[120]:


fig,ax = plt.subplots(figsize=(13,6))
ax.plot(df_week_lag,color="blue",label="Difference")
ax.plot(lag_roll_mean,color="red",label="Rolling Mean")
ax.plot(lag_roll_std,color="black",label="Rolling Standard Deviation")
ax.legend()
fig.tight_layout()


# ### 3. Log

# In[121]:


logged_week = np.log1p(df_week["Weekly_Sales"]).dropna() #taking log of data


# In[122]:


log_roll_mean = logged_week.rolling(window=2,center=False).mean()
log_roll_std = logged_week.rolling(window=2,center=False).std()


# In[123]:


fig,ax = plt.subplots(figsize=(13,6))
ax.plot(logged_week,color="blue",label="Difference")
ax.plot(log_roll_mean,color="red",label="Rolling Mean")
ax.plot(log_roll_std,color="black",label="Rolling Standard Deviation")
ax.legend()
fig.tight_layout()


# I tried my data without any changes, then tried with shifting, taking log and difference version of data. Differenece data give best results. So, I decided to take difference and use this data.

# ## Auto-ARIMA MODEL

# ### Train-Test Split

# In[124]:


train_data_diff = df_week_diff[:int(0.7*(len(df_week_diff)))]
test_data_diff = df_week_diff[int(0.7*(len(df_week_diff))):]


# In[125]:


model_auto_arima = auto_arima(train_data_diff,trace=True,start_p=0,start_q=0,
                             start_P=0,start_Q=0,max_p=20,max_q=20,max_P=20,max_Q=20,
                             seasonal=True,maxiter=200,information_criterion="aic",stepwise=False,
                             suppress_warnings=True,D=1,max_D=10,error_action="ignore",
                             approximation=False)

model_auto_arima.fit(train_data_diff)


# In[126]:


y_pred = model_auto_arima.predict(n_periods=len(test_data_diff))
y_pred = pd.DataFrame(y_pred,index=test_data.index,columns=["Prediction"])
plt.figure(figsize=(20,6))
plt.title("Prediction pf Weekly Sales Using Auto-ARIMA",fontsize=20)
plt.plot(train_data_diff,label="Train")
plt.plot(test_data_diff,label="Test")
plt.plot(y_pred,label="Prediction of ARIMA")
plt.legend(loc="best")
plt.xlabel("Date",fontsize=14)
plt.ylabel("Weekly Sales",fontsize=14)
plt.show()


# I do not like the pattern of predictions so I decided to try another model.

# ## ExponentialSmoothing

# Exponential Smooting are used when data has trend, and it flattens the trend.My difference data has some minus and zero values, so I used additive seasonal and trend instead of multiplicative. Seasonal periods are chosen from the decomposed graphs above. For tuning the model with iterations take too much time so, I changed and tried model for different parameters and found the best parameters and fitted them to model.

# In[127]:


model_holt_winters = ExponentialSmoothing(train_data_diff,seasonal_periods=20,seasonal="additive",
                                         trend="additive",damped=True).fit() #Taking additive trend and seasonality
y_pred = model_holt_winters.forecast(len(test_data_diff)) #Predict the test data

#Visualize train, test and predicted data
plt.figure(figsize=(20,6))
plt.title('Prediction of Weekly Sales using ExponentialSmoothing', fontsize=20)
plt.plot(train_data_diff, label='Train')
plt.plot(test_data_diff, label='Test')
plt.plot(y_pred, label='Prediction using ExponentialSmoothing')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.show()


# In[128]:


wmae_test(test_data_diff, y_pred)


# At the end, I found best results for my data with Exponential Smoothing Model.
