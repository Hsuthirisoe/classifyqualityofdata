#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:30:18 2019

@author: HTS
"""



import os

import cv2     
import math   
import matplotlib.pyplot as plt  
from glob import glob
import pandas as pd

import numpy as np    
from keras.utils import np_utils
from skimage.transform import resize  


count = 0;
img_array=[]
videos = 'videos/*.avi'
img_array = glob(videos)
for x in img_array:
 #if x == 3 break:   
 vidcap = cv2.VideoCapture(x)
 
 success,image = vidcap.read()

 while success:
   

  
   cv2.imwrite("images/frame%d.jpg" % count, image)
   success,image = vidcap.read()     
   if cv2.waitKey(10) == 27:                    
      break
  
   count += 1



data = pd.read_csv('mapping.csv')     
data.head()   

   
X = [ ]     
for img_name in data.Image_ID:
    img = plt.imread('images/' + img_name)
    X.append(img)  
X = np.array(X)   
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224), mode='constant').astype(int)     
    image.append(a)
X = np.array(image)
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')     
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    


#print ("Doneq!")
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout


#print ("Doneeee")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    
#print ("Doneeeeee")
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape


#print ("Doeeeewewne")

X_train = X_train.reshape(208, 7*7*512)     
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()     
X_valid = X_valid/X_train.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))   
model.add(Dense(units=1024, activation='sigmoid')) 
model.add(Dense(3, activation='softmax'))   

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history= model.fit(train, y_train, epochs=250, validation_data=(X_valid, y_valid))

"""
print (history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""

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
   

  
  
   cv2.imwrite("images/test%d.jpg" % count, image)
   success,image = vidcap.read()     
   if cv2.waitKey(10) == 27:                    
      break
  
   count += 1



data = pd.read_csv('test.csv')    
data.head()   

   
X = [ ]    
for img_name in data.Image_ID:
    img = plt.imread('images/' + img_name)
    X.append(img) 
X = np.array(X)    

 # one hot encoding Classes
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224), mode='constant').astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      
# preprocessing the input data
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


test_image = test_image/test_image.max()




df = pd.read_csv('test copy.csv')
df.head()
print (df.head())
thresh = 0.5
df['predicted_RF'] = (df.test >= 0.5).astype('int')
df['predicted_LR'] = (df.train >= 0.5).astype('int')
print(df.head())


predictions = model.predict_classes(test_image)
print("The screen time of good frame is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of unclear frame is", predictions[predictions==2].shape[0], "seconds")

num1=predictions[predictions==1].shape[0]
num2=predictions[predictions==2].shape[0]
if (num1 >= num2) :
   print("This video file is able to use")
else:
   
   print("This video file include a lot of unnecessary frames")

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0 
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    
    ACC = (TP+TN)/(TP+FP+FN+TN)
    recall = (TP)/(TP+FN)
    precision = (TP)/ (TP+FP)
    fscore = (2 * recall * precision) / (recall + precision)  

    return {'TP':TP, 'FP':FP, 'ACC':ACC , 'recall':recall, 'precision':precision, 'fscore':fscore}
print('TP:',perf_measure(df.actual_label.values, df.predicted_RF.values))
#perf_measure(df.actual_label.values, df.predicted_RF.values)
#isinstance (perf_measure(df.actual_label.values, df.predicted_RF.values))


 #, 
a=perf_measure(df.actual_label.values, df.predicted_RF.values)
FP= (a['TP'])
TP= (a['FP'])

 
plt.plot(FP, TP, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate',fontsize=17)
plt.ylabel('True Positive Rate',fontsize=17)
plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize=17)
plt.legend()
plt.show()


