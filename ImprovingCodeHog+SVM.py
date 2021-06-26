# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:14:59 2021

@author: Med Chlif
"""

import numpy as np 
import matplotlib.pyplot as plt 
import glob
import cv2
import os
from skimage.feature import hog

import seaborn as sns



print(os.listdir("dataset"))

train_images = []
train_labels = []

X= []


for directory_path in glob.glob("dataset/training_set/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    # label = []
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        train_images.append(fd)
        X_train=np.vstack(train_images).astype(np.float64)
        train_labels.append(label)
        
# from keras.utils import to_categorical

# train_labels2 = to_categorical(train_labels)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_labels)

y_train = le.transform(train_labels).astype(np.float64)



#####################TEST DIRECTORY ################################



test_images = []
test_labels = [] 

for directory_path in glob.glob("dataset/test_set/*"):
    trining_labels = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        test_images.append(fd)
        X_test=np.vstack(test_images).astype(np.float64)
        test_labels.append(trining_labels)

le.fit(test_labels)
y_test = le.transform(test_labels).astype(np.float64)


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model = LinearSVC()

svc_model.fit(X_train,y_train)

y_predict=svc_model.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix

cm = confusion_matrix(y_test,y_predict)
        
sns.heatmap(cm, annot=True,fmt='d')


print(classification_report( y_test, y_predict))

















