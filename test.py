# coding: utf-8
# In[2]:
# -*- coding: UTF-8 -*
from tensorflow import keras
import numpy as np
import time
category=['Cat','Dog']
model = keras.models.load_model('./weight2/weights.hdf5') # return a Model instance
print('model load successfully')
path = './18.JPG'

# get the running time
now=time.time()
image = keras.preprocessing.image.load_img(path, target_size=(224, 224))
image = keras.preprocessing.image.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
result = model.predict(image)
end = time.time()
print('Using Time : {:.3f}s'.format(end - now)) # print the running time
label = np.argmax(result[0])
print('predict animal is :', category[label])
print('probaility is ', max(result[0]))