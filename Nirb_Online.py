# -*- coding: utf-8 -*-
## NIRB script python Online part 

## Elise Grosjean
## 01/2021

import os
import os.path as osp
import pickle

from BasicTools.FE import FETools as FT
import Readers as MR

import SolutionVTKWriter as SVTKW

import numpy as np
import sys

## Directories
currentFolder=os.getcwd()

CoarseData=osp.join(currentFolder,'Coarse')
FineData=osp.join(currentFolder,'Fine')

ns=0 #number of snapshots
for _, _, files in os.walk(FineData): #compte le nombre de fichiers dans FineData
    ns=len(files)
print("number of snapshots: ",ns)

dimension=3
RectificationPT=True ##rectification Post-treatment
print("-----------------------------------")
print(" STEP2: start Online nirb          ")
print("-----------------------------------")

    ##################################################
    # LOAD DATA FOR ONLINE
    ##################################################

# retrieve the reduced basis
inputName="reducedOrderBasisU.pkl"
reducedOrderBasisU=pickle.load(open(inputName, "rb"))

nev=np.shape(reducedOrderBasisU)[0]
# retrieve the mass and rigidity matrices or construct them after 
inputName="l2ScalarProducMatrix.pkl"
l2ScalarProducMatrix=pickle.load(open(inputName, "rb"))
inputName="h1ScalarProducMatrix.pkl"
h1ScalarProducMatrix=pickle.load(open(inputName, "rb"))


## READ FINE MESH
meshFileName=FineData+"/snapshot_0.vtu"
Finemesh = MR.Readmesh(meshFileName)
#Finemeshmesh.GetInternalStorage().nodes = Finemesh.GetInternalStorage().nodes[:,:2] #CAS 2D
print("Fine Mesh defined in " + meshFileName + " has been read")

nbeOfComponentsPrimal = 1 #nbre de composants du champ
numberOfNodes = Finemesh.GetNumberOfNodes()
print("nbNodes",numberOfNodes)

## Coarse mesh
meshFileName2=FineData+"/snapshot_0.vtu"
Coarsemesh = MR.Readmesh(meshFileName2)
#Coarsemeshmesh.GetInternalStorage().nodes = Coarsemesh.GetInternalStorage().nodes[:,:2] #CAS 2D
print("Coarse Mesh defined in " + meshFileName2 + " has been read")

nbeOfComponentsPrimal = 3 #nbre de composants du champ
numberOfNodes2 = Coarsemesh.GetNumberOfNodes()
print("nbNodes",numberOfNodes2)

"""
print("ComputeL2ScalarProducMatrix and H1ScalarProducMatrix with BasicTools...")
from scipy import sparse
#l2ScalarProducMatrix=sparse.eye(numberOfNodes*nbeOfComponentsPrimal)

l2ScalarProducMatrix = FT.ComputeL2ScalarProducMatrix(Finemesh, nbeOfComponentsPrimal)
h1ScalarProducMatrix = FT.ComputeH10ScalarProductMatrix(Finemesh, nbeOfComponentsPrimal)
"""

#coarse solution
CoarseSnapshot =MR.VTKReadToNp("Velocity",CoarseData+"/snapshot_",ns-1)

#interpolation (nearest)

inputmesh=Coarsemesh
inputnodes=inputmesh.nodes
outputmesh=Finemesh
outputnodes=outputmesh.nodes
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse import coo_matrix 

kdt = cKDTree(inputnodes)
nbtp = outputnodes.shape[0]
_, ids = kdt.query(outputnodes)
cols=ids
row = np.arange(nbtp)
data = np.ones(nbtp)
Operator=coo_matrix((data, (row, cols)), shape=(nbtp , inputnodes.shape[0]))

#Compute the projected data using the projection operator
CoarseInterpolatedSnapshot = Operator.dot(CoarseSnapshot)
CoarseInterpolatedSnapshot=CoarseInterpolatedSnapshot.flatten()

#CoarseInterpolatedSnapshot=VTKSR.VTKReadToNp("Velocity",CoarseData+"/snapshot_",ns-1).flatten()

    ##################################################
    # ONLINE COMPRESSION
    ##################################################
print("-----------------------------------")
print(" STEP3: Snapshot compression       ")
print("-----------------------------------")
CompressedSolutionU= CoarseInterpolatedSnapshot@(l2ScalarProducMatrix@reducedOrderBasisU.transpose())

if RectificationPT==True:
    # retrieve the reduced basis
    inputName="rectification.pkl"
    R=pickle.load(open(inputName, "rb"))
    coef=np.zeros(nev)
    for i in range(nev):
        coef[i]=0
        for j in range(nev):
            coef[i]+=R[i,j]*CompressedSolutionU[j]
        
    #print("coef without rectification: ", CompressedSolutionU[0])
    #print("coef with rectification ", coef)

    reconstructedCompressedSolution = np.dot(coef, reducedOrderBasisU) #with rectification

else:
    reconstructedCompressedSolution = np.dot(CompressedSolutionU, reducedOrderBasisU)
    ##################################################
    # SAVE APPROXIMATION TO VTK
    ##################################################

reconstructedCompressedSolution=reconstructedCompressedSolution.reshape(numberOfNodes,nbeOfComponentsPrimal)

VTKBase = MR.VTKReadmesh(FineData+"/snapshot_0.vtu")
SVTKW.numpyToVTKWrite(VTKBase,reconstructedCompressedSolution,"approximation"+str(nev)+".vtu")

reconstructedCompressedSolution=reconstructedCompressedSolution.flatten()

    ##################################################
    # ONLINE ERRORS
    ##################################################
print("-----------------------------------")
print(" STEP4: L2 and H1 errors           ")
print("-----------------------------------")

print("reading exact solution...")
exactSolution=MR.VTKReadToNp("Velocity",FineData+"/snapshot_",ns-1).flatten()
compressionErrors=[]
H1compressionErrors=[]
norml2ExactSolution=np.sqrt(exactSolution@(l2ScalarProducMatrix@exactSolution))
normh1ExactSolution=np.sqrt(exactSolution@(h1ScalarProducMatrix@exactSolution))

if norml2ExactSolution != 0:
    t=reconstructedCompressedSolution-exactSolution
    relError=np.sqrt(t@l2ScalarProducMatrix@t)/norml2ExactSolution
    relH1Error=np.sqrt(t@h1ScalarProducMatrix@t)/normh1ExactSolution
   
else:
    relError = np.linalg.norm(reconstructedCompressedSolution-exactSolution)
H1compressionErrors.append(relH1Error)
compressionErrors.append(relError)
print("H1 NIRB error =", H1compressionErrors)
print("L2 NIRB error =", compressionErrors)

print("-----------------------------------")
print(" ONLINE STAGE DONE                 ")
print("-----------------------------------")
