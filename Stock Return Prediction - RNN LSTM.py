#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# In[2]:


os.chdir('C://Users//User//Desktop//')
print (os.getcwd())


# In[3]:


dataset_train = pd.read_csv('Final_Data_collected.csv')
dataset_train.dropna(how='any', inplace=True)
dataset_train.head()


# In[4]:


training_set=dataset_train.iloc[:,1:2].values
training_set


# In[5]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_fitter=sc.fit(training_set)
training_set_scaled=training_fitter.transform(training_set)


# In[6]:


X_train = []
y_train = []


# In[7]:


for i in range(60, 2258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])


# In[8]:


X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape


# In[9]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[11]:


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (X_train.shape[1],1)))


# In[12]:


regressor.add(Dropout(0.2))


# In[13]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[14]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[15]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[16]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[17]:


regressor.add(Dense(units = 1))


# In[18]:


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])


# In[19]:


model=regressor.fit(X_train,y_train,epochs = 100, batch_size = 16)
model


# In[20]:


dataset_test = pd.read_csv('Final_Data_collected.csv')
dataset_test.dropna(inplace=True)
dataset_test.head()


# In[21]:


real_stock_price = dataset_test.iloc[:,1:2].values


# In[22]:


real_stock_price


# In[23]:


dataset_total = pd.concat((dataset_train.iloc[:,1],dataset_test.iloc[:,1]),axis = 0)
dataset_total


# In[24]:


inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs


# In[25]:


inputs = inputs.reshape(-1,1)
inputs


# In[26]:


inputs=training_fitter.transform(inputs)
inputs.shape


# In[27]:


x_test = []
for i in range(60,2258):
    x_test.append(inputs[i-60:i,0])


# In[28]:


x_test = np.array(x_test)
x_test.shape


# In[29]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[30]:


predicted_price = regressor.predict(x_test)
predicted_price


# In[31]:


predicted_price = training_fitter.inverse_transform(predicted_price)
predicted_price


# In[32]:


plt.figure(figsize=(10,5))
plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
#plt.legend()
plt.show()


# In[34]:


print(max(model.history['accuracy'])*100)

