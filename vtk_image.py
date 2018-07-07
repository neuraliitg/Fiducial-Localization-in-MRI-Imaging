import vtk
from vtk.util import numpy_support 
import os
import numpy
from matplotlib import pyplot, cm
import plotly.plotly as py
import plotly.graph_objs as go

PathDicom = "./Sampdata/SE5/"
reader  = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()

_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1,_extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

ConstPixelSpacing = reader.GetPixelSpacing()

print(ConstPixelDims)
print(ConstPixelSpacing)

shiftScale = vtk.vtkImageShiftScale()
shiftScale.SetScale(reader.GetRescaleSlope())
shiftScale.SetShift(reader.GetRescaleOffset())
shiftScale.SetInputConnection(reader.GetOutputPort())
shiftScale.Update()

print(reader.GetRescaleSlope())
print(reader.GetRescaleOffset())

# Get the 'vtkImageData' object from the reader
imageData = reader.GetOutput()
# Get the 'vtkPointData' object from the 'vtkImageData' object
pointData = imageData.GetPointData()
# Ensure that only one array exists within the 'vtkPointData' object
assert (pointData.GetNumberOfArrays()==1)
# Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
arrayData = pointData.GetArray(0)
# Convert the `vtkArray` to a NumPy array
ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
# Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

trace = go.Heatmap(z = numpy.rot90(ArrayDicom[:,120,:]))
data = [trace]
py.iplot(data, filename = 'basic-heatmap')

threshold = vtk.vtkImageThreshold()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(400)
threshold.ReplaceInOn()
threshold.SetInValue(0)
threshold.ReplaceOutOn()
threshold.SetOutValue(1)
threshold.Update()
