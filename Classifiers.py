# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:08:03 2017

@author: kmcfall, iwatts
"""
import numpy as np

def featEdge(ii,isVert,rowUL,colUL,w,h,im=None):
    # ii: integral image from which to calculate feature
    # isVert: boolean for whether feature has left/right (True) or top/bottom (False) halves
    # rowUL: upper left hand corner row    position for feature
    # colUL: upper left hand corder column position for feature
    # w: width of feature  - NOTE: must be multiple of 2 if isVert = True
    # h: height of feature - NOTE: must be multiple of 2 if isVert = False
    # im: image to draw the feature on for visualization. Skip drawing if None.
    if isVert: # Is left/right halves
        # Implement Figure 2 where area D is split in left/right halves
        one    = ii[rowUL  , colUL         ]
        two    = ii[rowUL  , colUL+int(w/2)]
        three  = ii[rowUL+h, colUL         ]
        four   = ii[rowUL+h, colUL+int(w/2)]
        Dleft  = four + one - (two + three) # Pixel sum in left half of D
        
        one    = two
        three  = four
        two    = ii[rowUL  , colUL+w]
        four   = ii[rowUL+h, colUL+w]
        Dright = four + one - (two + three) # Pixel sum in right half of D
        
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+h, colUL:colUL+int(w/2)  ] = 255
            im[rowUL:rowUL+h, int(w/2)+colUL:colUL+w] =   0
    else:
        # Implement Figure 2 where area D is split in top/bottom halves
        one    = ii[rowUL         , colUL  ]
        two    = ii[rowUL         , colUL+w]
        three  = ii[rowUL+int(h/2), colUL  ]
        four   = ii[rowUL+int(h/2), colUL+w]
        Dleft  = four + one - (two + three) # Pixel sum in top half of D
        
        one    = three
        two    = four
        three  = ii[rowUL+h  , colUL  ]
        four   = ii[rowUL+h  , colUL+w]
        Dright = four + one - (two + three) # Pixel sum in right half of D
        
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+int(h/2),   colUL:colUL+w] = 255
            im[rowUL+int(h/2):rowUL+h, colUL:colUL+w] =   0
    return Dleft - Dright # Return difference between left and right halves

def featLine(ii,isVert,rowUL,colUL,w,h,im=None):
    # ii: integral image from which to calculate feature
    # isVert: boolean for whether feature has left/right (True) or top/bottom (False) halves
    # rowUL: upper left hand corner row    position for feature
    # colUL: upper left hand corder column position for feature
    # w: width of feature  - NOTE: must be multiple of 3 if isVert = True
    # h: height of feature - NOTE: must be multiple of 3 if isVert = False
    # im: image to draw the feature on for visualization. Skip drawing if None.
    if isVert: # Is left/middle/right thirds
        # Implement Figure 2 where area D is split in left/middle/right thirds
        one    = ii[rowUL  , colUL         ]
        two    = ii[rowUL  , colUL+int(w/3)]
        three  = ii[rowUL+h, colUL         ]
        four   = ii[rowUL+h, colUL+int(w/3)]
        Dleft  = four + one - (two + three) # Pixel sum in left third of D
        
        one    = two
        three  = four
        two    = ii[rowUL  , colUL+int(2*w/3)]
        four   = ii[rowUL+h, colUL+int(2*w/3)]
        Dmid = four + one - (two + three) # Pixel sum in middle third of D
        
        one    = two
        three  = four
        two    = ii[rowUL  , colUL+w]
        four   = ii[rowUL+h, colUL+w]
        Dright = four + one - (two + three) # Pixel sum in right third of D
        
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+h, colUL:colUL+int(w/3)           ] = 255
            im[rowUL:rowUL+h, int(w/3)+colUL:colUL+int(2*w/3)] =   0
            im[rowUL:rowUL+h, int(2*w/3)+colUL:colUL+w       ] = 255
    else:
        # Implement Figure 2 where area D is split in top/middle/bottom thirds
        one    = ii[rowUL         , colUL  ]
        two    = ii[rowUL         , colUL+w]
        three  = ii[rowUL+int(h/3), colUL  ]
        four   = ii[rowUL+int(h/3), colUL+w]
        Dleft  = four + one - (two + three) # Pixel sum in top third of D
        
        one    = three
        two    = four
        three  = ii[rowUL+int(2*h/3) , colUL  ]
        four   = ii[rowUL+int(2*h/3) , colUL+w]
        Dmid   = four + one - (two + three) # Pixel sum in middle third of D
        
        one    = three
        two    = four
        three  = ii[rowUL+h , colUL  ]
        four   = ii[rowUL+h , colUL+w]
        Dright = four + one - (two + three) # Pixel sum in bottom third of D
        
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+int(h/3)           , colUL:colUL+w] = 255
            im[int(h/3)+rowUL:rowUL+int(2*h/3), colUL:colUL+w] =   0
            im[int(2*h/3)+rowUL:rowUL+h       , colUL:colUL+w] = 255
    return Dleft + Dright - Dmid # Return difference between outside and middle thirds


def featChecker(ii,isVert,rowUL,colUL,w,h,im=None):
    # ii: integral image from which to calculate feature
    # rowUL: upper left hand corner row    position for feature
    # colUL: upper left hand corder column position for feature
    # w: width of feature  - NOTE: must be multiple of 2
    # h: height of feature - NOTE: must be multiple of 2
    # im: image to draw the feature on for visualization. Skip drawing if None.
    one    = ii[rowUL         , colUL         ]
    two    = ii[rowUL         , colUL+int(w/2)]
    three  = ii[rowUL+int(h/2), colUL         ]
    four   = ii[rowUL+int(h/2), colUL+int(w/2)]
    DUleft  = four + one - (two + three) # Pixel sum in upper left quarter of D
    
    one    = two
    three  = four
    two    = ii[rowUL         , colUL+w]
    four   = ii[rowUL+int(h/2), colUL+w]
    DUright = four + one - (two + three) # Pixel sum in upper right quarter of D
    
    one    = ii[rowUL+int(h/2)  , colUL         ]
    two    = ii[rowUL+int(h/2)  , colUL+int(w/2)]
    three  = ii[rowUL+h         , colUL         ]
    four   = ii[rowUL+h         , colUL+int(w/2)]
    DLleft  = four + one - (two + three) # Pixel sum in lower left quarter of D 
    
    one    = two
    three  = four
    two    = ii[rowUL+int(h/2)  , colUL+w]
    four   = ii[rowUL+h         , colUL+w]
    DLright = four + one - (two + three) # Pixel sum in lower right quarter of D
     
    if isVert == True:
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+int(h/2)  ,   colUL:colUL+int(w/2)  ] = 255
            im[rowUL:rowUL+int(h/2)  ,   int(w/2)+colUL:colUL+w] =   0
            im[int(h/2)+rowUL:rowUL+h,   colUL:colUL+int(w/2)  ] =   0
            im[int(h/2)+rowUL:rowUL+h,   int(w/2)+colUL:colUL+w] = 255

    else:
        if im is not None: # If desired to visualize feature
            im[rowUL:rowUL+int(h/2)  ,   colUL:colUL+int(w/2)  ] =   0
            im[rowUL:rowUL+int(h/2)  ,   int(w/2)+colUL:colUL+w] = 255
            im[int(h/2)+rowUL:rowUL+h,   colUL:colUL+int(w/2)  ] = 255
            im[int(h/2)+rowUL:rowUL+h,   int(w/2)+colUL:colUL+w] =   0
    
    return (DUleft + DLright) - (DLleft + DUright)  # Return difference
    

def getIntegralImage(i):
    ii = np.zeros_like(i,dtype='int64')
    s  = np.zeros_like(i,dtype='int64')
    # Implement Equations 1 and 2 and return integral image of input image
    for x in range(ii.shape[0]):
        s[x,0] = i[x,0]
        for y in range(ii.shape[1]):
            if y != 0:
                s[x,y] = s[x,y-1] + i[x,y]
            if x == 0:
                ii[x,y] = s[x,y]
            else:
                ii[x,y] = ii[x-1,y] + s[x,y]
    return ii

def makeWeakClassifierList():
    # Return list of all available weak classifiers
    weak = [] # Begin empty

    for w in (2,6,12,16):
        for h in (2,6,12,16):
            for col in range(0,19-w):
                for row in range(0,19-h):
                    # For this combination of h, w, rol, and col:
                    # Add edge features for vertical/horizontal and greater/less than
                    weak.append(weakClassifier('edge', True ,row,col,w,h,-1))
                    weak.append(weakClassifier('edge', True ,row,col,w,h, 1))
                    weak.append(weakClassifier('edge', False,row,col,w,h,-1))
                    weak.append(weakClassifier('edge', False,row,col,w,h, 1))
                
    for w in (3,9,15):
        for h in (3,9,15):
            for col in range(0,19-w):
                for row in range(0,19-h):
                    # For this combination of h, w, rol, and col:
                    # Add edge features for vertical/horizontal and greater/less than
                    weak.append(weakClassifier('line', True ,row,col,w,h,-1))
                    weak.append(weakClassifier('line', True ,row,col,w,h, 1))
                    weak.append(weakClassifier('line', False,row,col,w,h,-1))
                    weak.append(weakClassifier('line', False,row,col,w,h, 1))
                    
    for w in (2,4,6,8,12,14,16,18):
        for h in (2,4,6,8,12,14,16,18):
            for col in range(0,19-w):
                for row in range(0,19-h):
                    # For this combination of h, w, rol, and col:
                    # Add edge features for vertical/horizontal and greater/less than
                    weak.append(weakClassifier('checker', True ,row,col,w,h,-1))
                    weak.append(weakClassifier('checker', True ,row,col,w,h, 1))
    return weak


def getFeatures(weak, x):
    # For a list of weak classifiers, calculate the feature value for each
    # given an input integral image x
    feat = np.empty((len(weak),)) # Create 1D array with an element for each weak classifier
    for j in range(len(weak)): # Loop through each classifier
        feat[j] = weak[j].getFeatureValue(x) # Calculate the jth feature value
    return feat # Return 1D array of features for integral image x

def errorFunc(w,h,label):
    if h == label:
        return 0
    else:
        return w

class strongClassifier:
    # A strong classifier is a linear combination of many weak classifiers
    def __init__(self): # Initialize to be empty of classifiers
        self.weakList = [] # List containing weak classifiers
        self.alpha    = [] # List containing importance of each weak classifier
        
    def addWeakClassifier(self,weak,beta): # Add weak classifier to the list
        self.weakList.append(weak) # Add weak classifier to the list
        if beta < 0.000000001: # Zero beta gives infinite alpha so limit beta
            beta = 0.000000001
            
        self.alpha.append(np.log(1/beta)) # Add weak classifier importance
        
    def predict(self,iiList):
        # Given a list of integral images, return class prediction for each
        pred = np.zeros(len(iiList))
        
        for x in range(len(iiList)):
            numSum = 0
            denSum = 0

            for t in range(len(self.weakList)):
               h = self.weakList[t].predict(self.weakList[t].getFeatureValue(iiList[x]))
               numSum += self.alpha[t]*h
               denSum += self.alpha[t]
            pred[x] = numSum/denSum >= 0.5
            
        return pred
    
    def getPerformance(self,iiList,label):
        pred = self.predict(iiList) # Get prediction for each image in iiList
        indIs  = np.where(label==1)[0] # Indices which are     faces
        indNot = np.where(label==0)[0] # Indices which are not faces
        detect =     np.mean(pred[indIs ] == 1) # Detection rate (1 = all faces correctly detected)
        falseP = 1 - np.mean(pred[indNot] == 0) # False positive rate (0 = all non-faces rejected)
        correct = np.mean(pred == label) # Fraction of correct images overall
        return round(detect,2), round(falseP,2), round(correct,2) # Only keep 2 decimal places

class weakClassifier:
    # A weak classifier is one of the rectangle features in Figure 1
    def __init__(self,kind,orient,rowUL,colUL,w,h,parity):
        self.kind   = kind   # Edge, line, or checkboard features
        self.orient = orient # Left/right, top/bottom, diagonal up/down
        self.rowUL  = rowUL  # Row    for upper left corner of feature
        self.colUL  = colUL  # Column for upper left corner of feature
        self.w      = w      # Feature width
        self.h      = h      # Feature height
        self.parity = parity # Feature parity, i.e. greater or less than in the h_j(x) equation
        self.theta = 0       # Threshold theta in the h_j(x) equation
        
    def getFeatureValue(self,ii):
        # Return the feature value given an input integral image
        if self.kind == 'edge':
            return featEdge(ii,self.orient,self.rowUL,self.colUL,self.w,self.h)
        elif self.kind == 'line':
            return featLine(ii,self.orient,self.rowUL,self.colUL,self.w,self.h)
        elif self.kind == 'checker':
            return featChecker(ii,self.orient,self.rowUL,self.colUL,self.w,self.h)
        
    def optimizeTheta(self,f,label):
        # f: 1D array of feature values for each input image
        # label: 1D array of labels (0 or 1) for each input image
        # 
        # Optimize for the highest correct classification rate.
        # Loop through the range of feature values and find the value of theta
        # where classification is best. Set self.theta to the optimal threshold.
        
        bestPerformance = 0
        for theta in range(-50,51):
            correctCount = 0
            for i in range(len(f)):
               if (self.parity*f[i] < self.parity*theta) == label[i]:
                   correctCount += 1
            newPerformance = correctCount
            if newPerformance > bestPerformance:
                bestPerformance = newPerformance
                self.theta = theta
        return None
    
    def getWeightedError(self,f,label,w):
        # f: 1D array of feature values for each input image
        # label: 1D array of labels (0 or 1) for each input image
        # w: weights indicating which input images require more effort since they were incorrect before
        #
        # Compute and return the weighted error measurement in Step 2 of Table 1.

        h = self.predict(f)
        error = 0
        
        for i in range(len(label)):
            error += errorFunc(w[i],h[i],label[i])
        return error

    
    
    
    def predict(self,f):
        # f is a 1D array of feature values for multiple images
        return self.parity*f < self.parity*self.theta # Equation for h_j(x)

