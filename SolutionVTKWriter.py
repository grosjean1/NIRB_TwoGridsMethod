# -*- coding: utf-8 -*-
## VTk writer with BasicTools

## Elise Grosjean
## 01/2021

import numpy as np
import vtk
from vtk.util import numpy_support
from BasicTools.Containers import Filters


def numpyToVTKWrite(VTKBase,Solution, solutionName="NIRBapproximation.vtu",dataName="Velocity"):

        numpy_array = Solution
        #print("shape vtkfile", np.shape(numpySnap_array))
        p = VTKBase.GetPointData()
        VTK_data = numpy_support.numpy_to_vtk(num_array=numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
        #size = VTK_data.GetSize()
        #print("size array", size)
        VTK_data.SetName(dataName)
        #name = VTK_data.GetName()
        p.AddArray(VTK_data)
        
        out_fname = solutionName

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(out_fname)
        writer.SetInputData(VTKBase)
        writer.SetDataModeToAscii()
        writer.Write()
        print('\nfile ', out_fname, ' written\n' )
