# -*- coding: utf-8 -*-
## NIRB script python Greedy Algorithm

## Elise Grosjean
## 01/2021


from BasicTools.FE import FETools as FT
import numpy as np
from scipy import linalg

##### ALGO GREEDY ######
def Greedy(snapshots,l2ScalarProducMatrix,h1ScalarProducMatrix,nev):
    """
    Greedy algorithm for the construction of the reduced basis
    orthogonal basis in H1 et L2 et orthonormalized in L2
    #Algo as in https://hal.archives-ouvertes.fr/hal-01897395
    """
    
    nbdegree=np.shape(l2ScalarProducMatrix)[0]
    ns=np.shape(snapshots)[0]
    print(nbdegree)
    print(np.shape(snapshots[0]))
    reducedOrderBasisU=np.zeros((nev,nbdegree)) #nev, nbd

    norm0=np.sqrt(snapshots[0]@l2ScalarProducMatrix@snapshots[0]) #norm L2 u0
    reducedOrderBasisU[0,:]=snapshots[0]/norm0 #first mode
    ListeIndex=[0] #first snapshot 

    basis=[]
    basis.append(np.array(snapshots[0]))
    
    for n in range(1,nev):
        #print("nev ",n)
        vecteurTest=dict() # dictionnary: vector in the reduced basis if maxTest if maximum
        for j in range(ns): 
            if not (j in ListeIndex): #if index not yet in the basis
                matVecProduct=snapshots[j]@(l2ScalarProducMatrix)
                coef=[matVecProduct@b for b in basis]
                w=snapshots[j]-sum((matVecProduct@b)/(b@l2ScalarProducMatrix@b)*b for b in basis)#potential vector to add in the reduced basis
                norml2=np.sqrt(w@(l2ScalarProducMatrix@w))
                normj=np.sqrt(matVecProduct@snapshots[j]) #norm L2 uj
                maxTest=norml2/normj #we seek the max
                vecteurTest[j]=[maxTest,w]
               
        ind=max(vecteurTest, key = lambda k: vecteurTest[k][0]) #index of the snapshot used
        print("index ",ind)
        ListeIndex.append(ind) #adding in the list
        norm=np.sqrt(vecteurTest[ind][1]@(l2ScalarProducMatrix@vecteurTest[ind][1]))
        basis.append(vecteurTest[ind][1])
        reducedOrderBasisU[n,:]=(vecteurTest[ind][1]/norm) #orthonormalization in L2

    ### H1 Orthogonalization
    K=np.zeros((nev,nev)) #rigidity matrix
    M=np.zeros((nev,nev)) #mass matrix
    for i in range(nev):
        matVecProduct=reducedOrderBasisU[i,:]@l2ScalarProducMatrix
        matVecRigProduct=reducedOrderBasisU[i,:]@h1ScalarProducMatrix
        for j in range(0,i+1):
            M[i, j] = matVecProduct.dot(reducedOrderBasisU[j,:])
            K[i,j] = matVecRigProduct.dot(reducedOrderBasisU[j,:])
    for j in range(nev):
        for i in range(j,nev):
            K[j,i]=K[i,j]
            M[j,i]=M[i,j]
            
    eigenValues,vr=linalg.eig(K, b=M) #eigenvalues + right eigenvectors
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = vr[:, idx]
    reducedOrderBasisU=np.dot(eigenVectors.transpose(),reducedOrderBasisU)

    for i in range(nev):
        reducedOrderBasisNorm=np.sqrt(reducedOrderBasisU[i,:]@(l2ScalarProducMatrix@reducedOrderBasisU[i,:]))
        reducedOrderBasisU[i,:]=np.divide(reducedOrderBasisU[i,:],reducedOrderBasisNorm) #L2 orthonormalization
    return reducedOrderBasisU
