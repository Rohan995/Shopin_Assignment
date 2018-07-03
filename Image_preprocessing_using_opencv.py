
# coding: utf-8

# In[1]:


import numpy as np
import urllib
import cv2
import matplotlib.pyplot as plt
 


# In[2]:


def url_to_image(url):
 
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 
	return image


# In[3]:


img=url_to_image('https://assets.myntassets.com/h_640,q_90,w_480/v1/assets/images/2354027/2018/1/16/11516097831111-MsTaken-Women-Navy-Blue-Striped-Top-4731516097830960-1.jpg')


# In[4]:


cv2.imwrite('07.jpg',img)


# In[5]:


cv2.imshow("Image", img)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


height, width, channels = img.shape
print( height, width, channels)


# In[7]:


resized_image = cv2.resize(img, (224, 224)) 
cv2.imshow("Image",resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


crop_img = img[115:579, 60:420]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


# In[ ]:




