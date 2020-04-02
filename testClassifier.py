# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:36:44 2017

@author: iwatts
"""

import pickle
import Classifiers
import glob
import cv2
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