#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:30:18 2019

@author: HTS
"""



import os
os.environ ['TF_CPP_MIN_LOG_LEVEL']= '2'
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
from glob import glob
import pandas as pd
#from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
#from skimage.trasform import rescale



#split video

count = 0;
img_array=[]
videos = 'videos/*.avi'
img_array = glob(videos)
for x in img_array:
 #if x == 3 break:   
 vidcap = cv2.VideoCapture(x)
 
 success,image = vidcap.read()

 while success:
   

   #print("THE IMAGE NAME IS ", x , ".JPG")
  
   cv2.imwrite("images/frame%d.jpg" % count, image)
   success,image = vidcap.read()     
   if cv2.waitKey(10) == 27:                    
      break
  
   count += 1



data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()   

   # printing first five rows of the file
X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('images/' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224), mode='constant').astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      # preprocessing the input data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set


#print ("Doneq!")
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout


#print ("Doneeee")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
#print ("Doneeeeee")
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape


#print ("Doeeeewewne")

X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()
# i. Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer

model.summary()

# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# iii. Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))




########

count = 0;
img_array=[]
videos = 'videos/naito.avi'
img_array = glob(videos)
for x in img_array:
 #if x == 3 break:   
 vidcap = cv2.VideoCapture(x)
 
 success,image = vidcap.read()

 while success:
   

   #print("THE IMAGE NAME IS ", x , ".JPG")
  
   cv2.imwrite("images/test%d.jpg" % count, image)
   success,image = vidcap.read()     
   if cv2.waitKey(10) == 27:                    
      break
  
   count += 1



data = pd.read_csv('test.csv')     # reading the csv file
data.head()   

   # printing first five rows of the file
X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('images/' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array

 # one hot encoding Classes
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224), mode='constant').astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      # preprocessing the input data
#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set


#print ("Doneq!")
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout


#print ("Doneeee")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
#print ("Doneeeeee")
X_train = base_model.predict(X)

X_train.shape

print (X_train.shape)
test_image = X_train.reshape(101, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

#####33




predictions = model.predict_classes(test_image)
print("The screen time of good frame is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of unclear frame is", predictions[predictions==2].shape[0], "seconds")

num1=predictions[predictions==1].shape[0]
num2=predictions[predictions==2].shape[0]
if (num1 >= num2) :
   print("This video file is able to use")
else:
   
   print("This video file include a lot of unnecessary frames")





