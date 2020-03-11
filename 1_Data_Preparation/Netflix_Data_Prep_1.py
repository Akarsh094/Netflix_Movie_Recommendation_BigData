
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import re
import os
from scipy.sparse import csr_matrix


# In[2]:


#Load Text Dataset
df1 = pd.read_csv(os.getcwd() + '/Netflix_Data/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df2 = pd.read_csv(os.getcwd() + '/Netflix_Data/combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df3 = pd.read_csv(os.getcwd() + '/Netflix_Data/combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df4 = pd.read_csv(os.getcwd() + '/Netflix_Data/combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
#title = pd.read_csv(os.getcwd() + '/Netflix_Data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])

print('data loaded...')
# In[ ]:

df = df1
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)

df.index = np.arange(0,len(df))


# In[ ]:


df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()


# In[ ]:


movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))


# In[ ]:
df = df[pd.notnull(df['Rating'])]



df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000000, :])


# In[ ]:


df.to_csv(os.getcwd() + "/Netflix_Data/combined_data_final.csv")

