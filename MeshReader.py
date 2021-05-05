# -*- coding: utf-8 -*-
## VTK Reader with BasicTools

## Elise Grosjean
## 01/2021


import numpy as np
from pathlib import Path
import os

def Readmesh(meshFileName):
    """
    meshFileName : str (name of the VTK mesh file (.vtu))
    ----------
    Read a vtu mesh (the fine and coarse meshes)
    and return BasicToolsUnstructuredMesh Fine and coarse meshes 
    """  

    assert isinstance(meshFileName, str)
    suffix = str(Path(meshFileName).suffix)
    if suffix == ".vtu":  
        from BasicTools.IO.VtuReader import VtkToMesh as Read
        from BasicTools.IO.VtuReader import LoadVtuWithVTK 

    else: 
        raise ("FileName error!")

    mesh = Read(LoadVtuWithVTK(meshFileName))
    #print(mesh)
    return mesh
  

def VTKReadmesh(meshFileName):
    """
    meshFileName : str (name of the VTK mesh file (.vtu))
    ----------
    return a vtu mesh (the fine and coarse meshes)
    """ 
    assert isinstance(meshFileName, str)
    suffix = str(Path(meshFileName).suffix)
    if suffix == ".vtu": 
        from BasicTools.IO.VtuReader import VtkToMesh as Read
        from BasicTools.IO.VtuReader import LoadVtuWithVTK 

    else: 
        raise ("FileName error!")
    mesh = LoadVtuWithVTK(meshFileName)
    #print(mesh)
    return mesh
  
