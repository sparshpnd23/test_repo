#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd


# In[128]:


df = pd.read_csv('wines.csv')


# In[129]:


y = df['Class']


# In[130]:


y_cat = pd.get_dummies(y)


# In[131]:


X = df.drop('Class' , axis=1)


# In[132]:


from keras.models import Sequential


# In[133]:


model  =  Sequential()


# In[134]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_cat,test_size=0.3,random_state=50)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)


# In[135]:


from keras.layers import Dense


# In[136]:


model.add(Dense(units=5 , input_shape=(13,), 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[137]:


model.add(Dense(units=8 , 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[138]:


model.add(Dense(units=2, 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[139]:


model.add(Dense(units=3, activation='softmax'))


# In[140]:


from keras.optimizers import RMSprop


# In[141]:


model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[142]:


accuracy = model.fit(X_train,y_train,epochs=20)


# In[143]:


X_test=sc.transform(X_test)
y_pred=model.predict(X_test)


# In[144]:


model.save('modelsave.h5')


# In[145]:


acc=accuracy.history['accuracy'][-1:][0]


# In[146]:


print(acc)


# In[ ]:





# In[ ]:





# In[ ]:




