
# coding: utf-8

# In[3]:


import numpy as np
import os
from skimage.viewer import ImageViewer
from skimage import data, io


# In[5]:


image= io.imread('https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2354027/2018/1/16/11516097831111-MsTaken-Women-Navy-Blue-Striped-Top-4731516097830960-1.jpg')
io.imshow(image)


# In[6]:


height, width, channels = image.shape
print( height, width, channels)


# In[9]:


from skimage.transform import resize
resized_image = resize(image, (224, 224))
io.imshow(resized_image)


# In[10]:


crop_img = image[115:579, 60:420]
io.imshow(crop_img)


# In[16]:


io.imsave(crop_img,"crop_img_sklearn")


# In[19]:


from scipy.misc import imsave
imsave('crop_img_sklearn.png',crop_img)

