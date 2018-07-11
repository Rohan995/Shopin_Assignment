
# coding: utf-8

# In[1]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


# In[2]:


weights_path = '/home/ashutosh/Downloads/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150


# In[3]:


train_data_dir = '/home/ashutosh/Documents/keras/data/train'
validation_data_dir = '/home/ashutosh/Documents/keras/data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[4]:


model = applications.VGG16(weights='imagenet', include_top=False,input_shape = (150,150,3))
print('Model loaded.')


# In[5]:


top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))


# In[12]:


top_model.load_weights(top_model_weights_path)
#top_model.add(top_model)
#model = Model(inputs= model.input, outputs= top_model(model.output))
new_model = Sequential() #new model
for layer in model.layers: 
    new_model.add(layer)
new_model.add(top_model)
model=new_model


# In[13]:


for layer in model.layers[:25]:
    layer.trainable = False
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[14]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

