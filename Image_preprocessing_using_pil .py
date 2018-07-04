
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# Download the image

# In[2]:


import urllib
urllib.urlretrieve("https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2354027/2018/1/16/11516097831111-MsTaken-Women-Navy-Blue-Striped-Top-4731516097830960-1.jpg", "00000001.jpg")


# #Load,Read and Display

# In[3]:


from PIL import Image
import cStringIO
file = urllib.urlopen('https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2354027/2018/1/16/11516097831111-MsTaken-Women-Navy-Blue-Striped-Top-4731516097830960-1.jpg')
im = cStringIO.StringIO(file.read())
img = Image.open(im)
img.show()




# #Original height and width

# In[4]:


width, height = img.size
print("width "+str(width))
print("height "+str(height))


# #Resize and display image

# In[5]:


resize_img=img.resize((224,224),Image.ANTIALIAS)
resize_img.show()


# In[6]:


crop_img=img.crop((60, 115, 420, 579))
crop_img.show()
crop_img.save('crop_img.gif')

