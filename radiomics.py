#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
from pydicom import dcmread
import cv2


# In[56]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# In[86]:


ct_path = "G:/My Drive/Research/TCIA/manifest-1651798841270/COVID-19-NY-SBU/A000801/12-25-1900-NA-CHEST AP VIEWONLY-00860/1.000000-AP-96858/1-1.dcm"


# In[89]:


ct = dcmread(ct_path)


# In[85]:


ct


# In[90]:


ct_slice = ct.pixel_array


# In[91]:


ct_slice.shape


# In[92]:


dim1 = int(ct_slice.shape[0]/30)
dim2 = int(ct_slice.shape[1]/30)


# In[93]:


plt.imshow(ct_slice,cmap = plt.cm.gray)
plt.show()


# In[94]:


ct_slice_res = cv2.resize(ct_slice, dsize=(dim2, dim1), interpolation=cv2.INTER_CUBIC)


# In[95]:


plt.imshow(ct_slice_res,cmap = plt.cm.gray)
plt.show()


# In[96]:


ct_slice_res_example = ct_slice[260:265, 260:265]


# In[97]:


ct_slice_res_example


# In[98]:


plt.imshow(ct_slice_res_example,cmap = plt.cm.gray)
plt.show()


# In[99]:


# special functions for using pyradiomics
from SimpleITK import GetImageFromArray
import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor # This module is used for interaction with pyradiomic
import logging
logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)


# In[100]:


# Instantiate the extractor
texture_extractor = RadiomicsFeatureExtractor(verbose=False)
texture_extractor.disableAllFeatures()
_text_feat = {ckey: [] for ckey in texture_extractor.featureClassNames}
texture_extractor.enableFeaturesByName(**_text_feat)

print('Extraction parameters:\n\t', texture_extractor.settings)
print('Enabled filters:\n\t', texture_extractor.enabledImagetypes) 
print('Enabled features:\n\t', texture_extractor.enabledFeatures) 


# In[101]:


texture_extractor.settings['normalize'] = True


# In[102]:


import numpy as np # for manipulating 3d images
import pandas as pd # for reading and writing tables
import h5py # for reading the image files
import skimage # for image processing and visualizations
import sklearn # for machine learning and statistical models
import os # help us load files and deal with paths
from pathlib import Path # help manage files


# In[103]:


get_ipython().run_cell_magic('time', '', 'results = texture_extractor.execute(GetImageFromArray(ct_slice_res),\n                            GetImageFromArray((ct_slice_res>0).astype(np.uint8)))')


# In[104]:


df_results = pd.DataFrame([results]).T


# In[105]:


df_results.shape


# In[106]:


df_results


# In[82]:


df_results


# In[55]:


df_results


# In[27]:


import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


# In[28]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# In[29]:


imag_path = "G:/My Drive/Research/TCIA/manifest-1651798841270/COVID-19-NY-SBU/"


# In[30]:


imgs = []
for dirpath, subdirs, files in os.walk(imag_path):
    for x in files:
        if x.endswith(".dcm"):
            imgs.append(os.path.join(dirpath, x))


# In[31]:


dim1 = 500
dim2 = 500


# In[32]:


all_df = []
all_df_rgb = []
for i in imgs:
    ct = dcmread(i)
    ct_slice = ct.pixel_array
    #dim1 = int(ct_slice.shape[0]/20)
    #dim2 = int(ct_slice.shape[1]/20)
    ct_slice_res = cv2.resize(ct_slice, dsize=(dim2, dim1), interpolation=cv2.INTER_CUBIC)
    norm = cv2.normalize(ct_slice_res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    backtorgb = cv2.cvtColor(norm,cv2.COLOR_GRAY2RGB)
    all_df.append(ct_slice_res)
    all_df_rgb.append(backtorgb)


# In[33]:


x_train = np.array(all_df_rgb[0:35])
x_test = np.array(all_df_rgb[35:])

#x_train = all_df[0:35]
#x_test = all_df[35:]


# In[34]:


x_train.shape


# In[35]:


y_train= np.random.choice([0, 1], size=(35,), p=[1./3, 2./3])
y_test= np.random.choice([0, 1], size=(17,), p=[1./3, 2./3])


# In[36]:


model = keras.Sequential(
    [
     keras.Input(shape=(dim1,dim2,3)),
     layers.Conv2D(20, 4, activation='relu'),
     layers.Conv2D(10, 2, activation='relu'),
     layers.MaxPooling2D(),
     layers.Flatten(),
     layers.Dense(10),
    layers.Dense(5),
    ]
)


# In[37]:


model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(),
  metrics=['accuracy']
)


# In[38]:


model.fit(
        x_train, 
        y_train, 
        epochs=30, 
        batch_size=5, 
        validation_data = (x_test, y_test))


# In[40]:


plt.imshow(x_train[10])


# In[41]:


explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(
         x_train[10], 
         model.predict
)
plt.imshow(x_train[10])
image, mask = explanation.get_image_and_mask(
         model.predict(
              x_train[10].reshape((1,dim1,dim2,3))
         ).argmax(axis=1)[0],
         positive_only=True, 
         hide_rest=False)
plt.imshow(mark_boundaries(image, mask))


# In[ ]:




