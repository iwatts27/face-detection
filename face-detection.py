# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 16:52:02 2017

@author: iwatts
"""

import cv2
import matplotlib.pyplot as plot
import pickle
import Classifiers
import glob
import numpy as np

im     = []
iiList = []
label  = []
numNot = 200 # Number of non-face images to use
numYes = 200 # Number of     face images to use

strong = pickle.load(open("strongClassifier.pkl",'rb'))

filesNOT = glob.glob('../faces/test/non-face/*.pgm') # List of filenames of non-faces
filesYES = glob.glob('../faces/test/face/*.pgm')     # List of filenames of     faces
for count in range(numNot): # loop through desired number of non-faces
    i = cv2.imread(filesNOT[count],-1)   # Read image
    im.append(i)                         # Append image to list of images
    ii = Classifiers.getIntegralImage(i) # Calculate integral image
    iiList.append(ii)                    # Append ii to list of integral images
    label.append(0)      # Append label 0 = non-face
print('Loaded not images')
for count in range(numYes): # loop through desired number of faces
    i = cv2.imread(filesYES[count],-1)   # Read image
    im.append(i)                         # Append image to list of images
    ii = Classifiers.getIntegralImage(i) # Calculate integral image
    iiList.append(ii)                    # Append ii to list of integral images
    label.append(1)      # Append label 1 = face
print('Loaded is  images')
label = np.array(label) # Convert list to 1D array

print(strong.getPerformance(iiList,label))

predict = strong.predict(iiList)

iiList = []
coordList = []
result = []
granularity = 2
size = 48 #multiple of granularity


full = cv2.cvtColor(cv2.imread('McFall.jpg'), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(full, cv2.COLOR_RGB2GRAY)

for y in range(int(gray.shape[0]/granularity)-int(size/granularity)):
    for x in range(int(gray.shape[1]/granularity)-int(size/granularity)):
        i = cv2.resize(gray[y*granularity:y*granularity+size,x*granularity:x*granularity+size],(19,19))
        iiList.append(Classifiers.getIntegralImage(i))
        coordList.append([x*granularity,y*granularity])

result = strong.predict(iiList)

for x in range(len(result)):
    if result[x] == 1:
        cv2.rectangle(full,(coordList[x][0],coordList[x][1]),(coordList[x][0]+size,coordList[x][1]+size),[0,0,0])

plot.figure(1)
plot.imshow(full)
