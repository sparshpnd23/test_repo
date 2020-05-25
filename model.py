#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd


# In[26]:


df = pd.read_csv('wines.csv')


# In[27]:


y = df['Class']


# In[28]:


y_cat = pd.get_dummies(y)


# In[29]:


X = df.drop('Class' , axis=1)


# In[30]:


from keras.models import Sequential


# In[31]:


model  =  Sequential()


# In[32]:


from keras.layers import Dense


# In[33]:


model.add(Dense(units=5 , input_shape=(13,), 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[34]:


model.add(Dense(units=8 , 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[35]:


model.add(Dense(units=2, 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[36]:


model.add(Dense(units=3, activation='softmax'))


# In[37]:


from keras.optimizers import RMSprop


# In[38]:


model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[43]:


accuracy = model.fit(X,y_cat, epochs=100)


# In[50]:


model.save('modelsave.h5')


# In[51]:


acc=accuracy.history['accuracy'][-1:][0]


# In[52]:


print(acc)


# In[ ]:





# In[ ]:





# In[ ]:




