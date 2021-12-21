# -*- coding: utf-8 -*-
## NIRB script python Offline part 

## Elise Grosjean
## 01/2021
""" 
NIRB 2 grids method
Offline part with Greedy algorithm and rectification (RectificationPT=False otherwise)

"""
import os
import os.path as osp

from BasicTools.FE import FETools as FT
import Readers as MR
#from initCase import initproblem
import pickle
import SVD
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

### INIT ### 
## Directories
currentFolder=os.getcwd()
FineData=osp.join(currentFolder,'Fine')

## Script Files - Initiate data
externalFolder=osp.join(currentFolder,'External')  #BLACK-BOX SOLVER


## post-treatment
RectificationPT=True

dimension=3 ## spatial dimension
nbeOfComponentsPrimal = 3 #number of components of the field

#################################################"


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

## READ FINE MESH
meshFileName=FineData+"/snapshot_0.vtu"
Finemesh = MR.Readmesh(meshFileName)
print("Mesh defined in " + meshFileName + " has been read")

numberOfNodes = Finemesh.GetNumberOfNodes()
print("number of nodes",numberOfNodes)

snapshots = []

print("ComputeL2ScalarProducMatrix and H1ScalarProducMatrix with BasicTools...")
from scipy import sparse
#l2ScalarProducMatrix=sparse.eye(numberOfNodes*nbeOfComponentsPrimal)
l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix(Finemesh, nbeOfComponentsPrimal)
h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(Finemesh, nbeOfComponentsPrimal)

for i in range(ns):

    snapshot =MR.VTKReadToNp("Velocity",FineData+"/snapshot_",i).flatten()
    snapshots.append(snapshot)
    #print("snapshot defined in " + FineData + "snapshot_"+str(i)+" has been read")
  

""" POD ou Greedy """
print("-----------------------------------")
print(" STEP1: Offline                    ")
print("-----------------------------------")
nev=1
tol=1e-6
if len(sys.argv)>1:
    nev=int(sys.argv[1]) #11   #number of modes ( apriori) or with eigenvalues of POD close to 1
else:
    tol=1e-6

print("number of modes: ",nev)

##### ALGO GREEDY
reducedOrderBasisU=GD.Greedy(snapshots,l2ScalarProducMatrix,h1ScalarProducMatrix,nev,tol)
nev=np.shape(reducedOrderBasisU)[0]
print("number of modes : ", nev)

print("-----------------------------------")
print("---- RECTIFICATION POST-TREATMENT -")
print("-----------------------------------")
#provided several coarse snapshots

if RectificationPT==True:
    CoarseData=osp.join(currentFolder,'Coarse')
    ## READ COARSE MESH
    meshFileName=CoarseData+"/snapshot_0.vtu"
    Coarsemesh = MR.Readmesh(meshFileName)
    print("Mesh defined in " + meshFileName + " has been read")

    CoarseSnapshots=[]
    for i in range(ns):
        snapshotH =MR.VTKReadToNp("Velocity",CoarseData+"/snapshot_",i)#.flatten()
        #interpolation (nearest)

        inputmesh=Coarsemesh
        inputnodes=inputmesh.nodes
        outputmesh=Finemesh
        outputnodes=outputmesh.nodes
        from scipy.spatial import cKDTree
        #from scipy import sparse
        from scipy.sparse import coo_matrix 

        kdt = cKDTree(inputnodes)
        nbtp = outputnodes.shape[0]
        _, ids = kdt.query(outputnodes)
        cols=ids
        row = np.arange(nbtp)
        data = np.ones(nbtp)
        Operator=coo_matrix((data, (row, cols)), shape=(nbtp , inputnodes.shape[0]))

        #Compute the projected data using the projection operator
        CoarseInterpolatedSnapshot = Operator.dot(snapshotH)
        snapshotH=CoarseInterpolatedSnapshot.flatten()

        CoarseSnapshots.append(snapshotH)
    
    R=GD.Rectification(snapshots,CoarseSnapshots,reducedOrderBasisU,l2ScalarProducMatrix,nev)
    outputName = "rectification.pkl"
    output = open(outputName, "wb")
    pickle.dump(R, output)
    output.close()

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
