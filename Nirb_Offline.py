# -*- coding: utf-8 -*-
## NIRB script python Offline part 

## Elise Grosjean
## 01/2021

import os
import os.path as osp

from BasicTools.FE import FETools as FT
import MeshReader as MR
import SolutionReader as VTKSR

#from initCase import initproblem

import pickle
import POD
import Greedy as GD
import numpy as np
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy

"""
Create data (mesh1,mesh2,snapshots,uH) for Sorbonne usecase
"""
""" 
----------------------------
              generate snapshots
----------------------------
""" 
## Directories
currentFolder=os.getcwd()
FineData=osp.join(currentFolder,'Fine')

## Script Files - Initiate data
externalFolder=osp.join(currentFolder,'External')  #BLACK-BOX SOLVER


print("-----------------------------------")
print(" STEP0: start init                 ")
print("-----------------------------------")
#initproblem(dataFolder) #snapshots must be in .vtu format!
print("-----------------------------------")
print(" STEP0: snapshots generated        ")
print("-----------------------------------")

"""
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
                                      Definition and initialization of the problem
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
"""
ns=0 #number of snapshots
for _, _, files in os.walk(FineData): #number of files in FineData
    ns=len(files)
print("number of snapshots: ",ns)

dimension=3
           
## READ FINE MESH
meshFileName=FineData+"/snapshot_0.vtu"
Finemesh = MR.Readmesh(meshFileName)


print("Mesh defined in " + meshFileName + " has been read")

nbeOfComponentsPrimal = 3 #nbre de composants du champ
numberOfNodes = Finemesh.GetNumberOfNodes()
print("nbNodes",numberOfNodes)

snapshots = []

print("ComputeL2ScalarProducMatrix and H1ScalarProducMatrix with BasicTools...")
from scipy import sparse
#l2ScalarProducMatrix=sparse.eye(numberOfNodes*nbeOfComponentsPrimal)
#unstructuredMesh = Finemesh.GetInternalStorage()
l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix(Finemesh, nbeOfComponentsPrimal)
h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(Finemesh, nbeOfComponentsPrimal)

for i in range(ns):

    snapshot =VTKSR.VTKReadToNp("Velocity",FineData+"/snapshot_",i).flatten()
    snapshots.append(snapshot)
    #print("snapshot defined in " + FineData + "snapshot_"+str(i)+" has been read")
  

""" POD ou Greedy """
print("-----------------------------------")
print(" STEP1: Offline                    ")
print("-----------------------------------")
nev=1
tol=1e-4
if len(sys.argv)>1:
    nev=int(sys.argv[1]) #11   #number of modes ( apriori) or with eigenvalues of POD close to 1
else:
    NBdeg=numberOfNodes*nbeOfComponentsPrimal
    CorrMatrix=np.zeros((ns,ns))
    snapshots = np.array(snapshots)
    for i, snapshot1 in enumerate(snapshots):
        
        matVecProduct = l2ScalarProducMatrix.dot(snapshot1)
        for j in range(0,i+1):
            CorrMatrix[i, j] = matVecProduct.dot(snapshots[j])
    for j, snapshot1 in enumerate(snapshots):
        for i in range(j,ns):
            CorrMatrix[j,i]=CorrMatrix[i,j]
    nev,eigenvalues=POD.GetNev(CorrMatrix,tol) #if use of RIC

print("number of modes: ",nev)

##### ALGO GREEDY
reducedOrderBasisU=GD.Greedy(snapshots,l2ScalarProducMatrix,h1ScalarProducMatrix,nev)

print("-----------------------------------")
print(" STEP1:Reduced Basis created       ")
print("-----------------------------------")
### Offline Errors
print("Offline Errors")
compressionErrors=[]
h1compressionErrors=[]

for snap in snapshots:
    exactSolution =snap
    #print(np.shape(exactSolution))
    #print(np.shape(reducedOrderBasisU))
    CompressedSolutionU= exactSolution@(l2ScalarProducMatrix@reducedOrderBasisU.transpose())
    reconstructedCompressedSolution = np.dot(CompressedSolutionU, reducedOrderBasisU) #pas de tps 0
    
    norml2ExactSolution=np.sqrt(exactSolution@(l2ScalarProducMatrix@exactSolution))
    normh1ExactSolution=np.sqrt(exactSolution@(h1ScalarProducMatrix@exactSolution))
    if norml2ExactSolution !=0 and normh1ExactSolution != 0:
        t=reconstructedCompressedSolution-exactSolution
        relError=np.sqrt(t@l2ScalarProducMatrix@t)/norml2ExactSolution
        relh1Error=np.sqrt(t@h1ScalarProducMatrix@t)/normh1ExactSolution
    else:
        relError = np.linalg.norm(reconstructedCompressedSolution-exactSolution)
    compressionErrors.append(relError)
    h1compressionErrors.append(relh1Error)

print("L2 compression error =", compressionErrors)
print("H1 compression error =", h1compressionErrors)


### save reduced basis
outputName = "reducedOrderBasisU.pkl"
output = open(outputName, "wb")
pickle.dump(reducedOrderBasisU, output)
output.close()
    

##optional
### save mass and rigidity matrices

outputName = "l2ScalarProducMatrix.pkl"
output = open(outputName, "wb")
pickle.dump(l2ScalarProducMatrix, output)
output.close()
outputName = "h1ScalarProducMatrix.pkl"
output = open(outputName, "wb")
pickle.dump(h1ScalarProducMatrix, output)
output.close()

print("-----------------------------------")
print(" OFFLINE STAGE DONE                ")
print("-----------------------------------")
