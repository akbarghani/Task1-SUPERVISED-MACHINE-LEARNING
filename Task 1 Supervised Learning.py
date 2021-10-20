#!/usr/bin/env python
# coding: utf-8

# # Supervised Machine Learnig Task The Spark Foundation  (Task 1)
# 
# ## Author: Akbar Ghani 
# ## The Spark Foundation Grip 2021 september to October

# ## **Linear Regression a basic concept of machine learning with  library of Python Scikit Learn**
# Solving the problem of Simple Linear regression ScikitLearn Python
# 
# ### **Simple Linear Regression**
# The objective of the task is to predict the score of student according to the number of study hoursof the student
# 

# In[1]:


# Import the required library for performing Task
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sk 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:/Users/Faisal Khan/Desktop/Spark Intern/Task 1.csv")


# In[3]:


df


# In[4]:


# Checking the null values
df.isnull().sum()


# In[5]:


#Checking the data type of varibles
df.dtypes


# #### **Statistical Summary of data**

# In[6]:


df.describe()


# ### Correlation Between Hours and Scores

# In[7]:


df.corr()


# ### creating scatter plot # Plotting the distribution of scores
# 

# In[8]:


df.plot(x='Hours', y='Scores', style='o')  
fig, ax = plt.subplots(figsize = (10 , 6))
ax.scatter(df["Hours"] , df["Scores"])
ax.set_xlabel('Hours')
ax.set_ylabel('Scores')
ax.set_title('Hours vs Percentage') 
plt.show()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# ### ** data Processing and making suitable according to the task **
# 
# The next step is to divide the data into  (inputs) and (outputs).

# In[9]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# ## splitting the data using Scikit learn library

# In[10]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ## Splitting the data into trainig and test

# In[11]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training  Has been Complete ")


# ## Making the Regression Line

# In[12]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### **Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[13]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# ## Comparing Observed vs Predicted values
# 

# In[14]:



df = pd.DataFrame({'observed ': y_test, 'Predicted value': y_pred})  
df 


# ## checking the value a given value of hours 

# In[20]:


hours = 9.25 
own_pred = regressor.predict(X)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[6]))


# In[ ]:





# In[ ]:





# In[ ]:




