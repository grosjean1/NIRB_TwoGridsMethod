# -*- coding: utf-8 -*-
## NIRB script python Greedy Algorithm

## Elise Grosjean
## 01/2021


import numpy as np
#from scipy import linalg

def GetNev(CorrelationMatrix,tol): #SVD 
    
    eigenValues, eigenVectors = np.linalg.eigh(CorrelationMatrix, UPLO="L") #svd
    
    idx = eigenValues.argsort()[::-1] #sort the eigenvalues
    eigenValues = eigenValues[idx]
    print("eigenvalues ",eigenValues)
    nev = 0
    #print(eigenValues[0]/np.sum(eigenValues))
    bound = (1 - tol ) * np.sum(eigenValues) #Relative Information Content (RIC) close to 1
    #print(bound)
    temp = 0
    for e in eigenValues:
        temp += e
        #print("temp",temp)
        if temp < bound:
            nev += 1 
    return nev,eigenValues[0:nev]
