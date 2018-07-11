
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# Train_folder
# 

# In[4]:


for i in range(1000):
    cat = cv2.imread('/home/ashutosh/Documents/keras/all/train/cat.'+str(i)+'.jpg',1)
    #cv2.imwrite('/home/ashutosh/Documents/keras/data/train/cats/cat'+str(i)+'.jpg',cat)


# In[5]:


for j in range(1000):
    dog = cv2.imread('/home/ashutosh/Documents/keras/all/train/dog.'+str(j)+'.jpg',1)
    #cv2.imwrite('/home/ashutosh/Documents/keras/data/train/dogs/dog.'+str(j)+'.jpg',dog)


# Validation_folder

# In[8]:


for i in range(400):
    cat = cv2.imread('/home/ashutosh/Documents/keras/all/train/cat.'+str(i+1000)+'.jpg',1)
    #cv2.imwrite('/home/ashutosh/Documents/keras/data/validation/cats/cat'+str(i+1000)+'.jpg',cat)


# In[9]:


for j in range(400):
    dog = cv2.imread('/home/ashutosh/Documents/keras/all/train/dog.'+str(j+1000)+'.jpg',1)
    #cv2.imwrite('/home/ashutosh/Documents/keras/data/validation/dogs/dog.'+str(j+1000)+'.jpg',dog)


# In[10]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# In[11]:


img_width, img_height = 150, 150
train_data_dir = '/home/ashutosh/Documents/keras/data/train'
validation_data_dir = '/home/ashutosh/Documents/keras/data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[12]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[13]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[14]:


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[15]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[16]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[17]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[18]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[19]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[20]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

