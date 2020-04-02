# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:12:18 2017

@author: kmcfall, iwatts
"""

import cv2
import glob
import numpy as np
import pickle
import Classifiers # Load the file Classifiers.py like a module
import copy

if __name__== "__main__":
    
    # Initialize lists as empty before loading data into them
    
    
    #im     = []
    #iiList = []
    #label  = []
    #w      = []
    #numNot = 800 # Number of non-face images to use
    #numYes = 800 # Number of     face images to use
    
    #weakList = Classifiers.makeWeakClassifierList()    # Create the list of weak classifiers
    #print('Number of features: ',len(weakList))        # Number of features
    #feat  = np.empty((0,len(weakList)),dtype='int64')  # Initialize zero rows, a column for each feature
    #filesNOT = glob.glob('../faces/train/non-face/*.pgm') # List of filenames of non-faces
    #filesYES = glob.glob('../faces/train/face/*.pgm')     # List of filenames of     faces
    #for count in range(numNot): # loop through desired number of non-faces
    #    i = cv2.imread(filesNOT[count],-1)   # Read image
    #    im.append(i)                         # Append image to list of images
    #    ii = Classifiers.getIntegralImage(i) # Calculate integral image
    #    iiList.append(ii)                    # Append ii to list of integral images
    #    feat = np.vstack((feat,Classifiers.getFeatures(weakList,ii))) # Add features to feature matrix
    #    label.append(0)      # Append label 0 = non-face
    #    w.append(1/numNot/2) # Calculate initial weight evenly distributed
    #print('Loaded not images')
    #for count in range(numYes): # loop through desired number of faces
    #    i = cv2.imread(filesYES[count],-1)   # Read image
    #    im.append(i)                         # Append image to list of images
    #    ii = Classifiers.getIntegralImage(i) # Calculate integral image
    #    iiList.append(ii)                    # Append ii to list of integral images
    #    feat = np.vstack((feat,Classifiers.getFeatures(weakList,ii))) # Add features to feature matrix
    #    label.append(1)      # Append label 1 = face
    #    w.append(1/numYes/2) # Calculate initial weight evenly distributed
    #print('Loaded is  images')
    #label = np.array(label) # Convert list to 1D array
    
    # Save important variables to save time when executing this the next time
    #pickle.dump(iiList, open("iiList.pkl","wb")) # Next time just load this, don't calculate again
    #pickle.dump(feat,open("feat.pkl","wb"))
    #pickle.dump(label,open("label.pkl","wb"))
    #pickle.dump(w,open("w.pkl","wb"))
    
    iiList = pickle.load(open("iiList.pkl", 'rb'))
    feat = pickle.load(open("feat.pkl", 'rb'))
    label = pickle.load(open("label.pkl", 'rb'))
    w = pickle.load(open("w.pkl", 'rb'))
    
    #print("Optimizing Weak Classifiers")
    # Loop through the weak classifiers and optimize each of them
    #for j in range(len(weakList)):    
    #    weakList[j].optimizeTheta(feat[:,j],label,j)
    
    #pickle.dump(weakList, open("weakClassifier.pkl","wb")) # Save weak classifiers for next time
    
    weakList = pickle.load(open("weakClassifier.pkl", 'rb')) #takes about 2 hours to optimize with ~20000 features and 1600 images
    
    strong  = Classifiers.strongClassifier() # Create a strong classifier empty of weak classifiers
    numSamp = feat.shape[0] # Each row    in the feature matrix in a sample input image
    numFeat = feat.shape[1] # Each column in the feature matrix is a feature (i.e. weak classifier)
    t = -1
    detect  = 0
    falseP  = 1
    correct = 0
    
    while t<75 or correct < .95 or falseP > .001:
        t += 1
        winner = 0
        lowestError = 0.5
        wsum = sum(w)
        w[:] = [x / wsum for x in w]
        print(sum(w))
        print(t)
        
        for j in range(numFeat):
            error = weakList[j].getWeightedError(feat[:,j],label,w)
            if error < lowestError:
                lowestError = error
                winner = j
        print(lowestError)
        print(winner)
        
        beta = lowestError / (1 - lowestError)
        
        for i in range(numSamp):
            if weakList[winner].predict(feat[i,winner]) != label[i]:
                w[i] = w[i] * 1 
            else:
                w[i] = w[i] * beta
                
        strong.addWeakClassifier(copy.deepcopy(weakList[winner]),beta)
        
        detect, falseP, correct = strong.getPerformance(iiList,label)
        print(strong.getPerformance(iiList,label))
        
    print(strong.getPerformance(iiList,label)) # Print performance in the training set
    # Save strong classifiers so they can be loaded to evaluate their performance
    pickle.dump(strong, open("strongClassifier.pkl","wb"))



