# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import os


def VTKReadToNp(SolutionName, tmpbaseFile, i):
    from BasicTools.IO.VtuReader import LoadVtuWithVTK
    from vtk.numpy_interface import dataset_adapter as dsa
        
    data = LoadVtuWithVTK(tmpbaseFile + str(i) + ".vtu")
        
    npArray = dsa.WrapDataObject(data).GetPointData().GetArray(SolutionName)
    return npArray

