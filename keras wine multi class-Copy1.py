#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('wines.csv')


# In[5]:


y = df['Class']


# In[6]:


y_cat = pd.get_dummies(y)


# In[7]:


X = df.drop('Class' , axis=1)


# In[8]:


from keras.models import Sequential


# In[9]:


model  =  Sequential()


# In[10]:


from keras.layers import Dense


# In[11]:


model.add(Dense(units=5 , input_shape=(13,), 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[12]:


model.add(Dense(units=8 , 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[13]:


model.add(Dense(units=2, 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[14]:


model.add(Dense(units=3, activation='softmax'))


# In[27]:


from keras.optimizers import RMSprop


# In[28]:


model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[32]:


model.fit(X,y_cat, epochs=100)


# In[35]:


model.get_weights()


# In[36]:


model.save('modelsave.h5')


# In[ ]:




