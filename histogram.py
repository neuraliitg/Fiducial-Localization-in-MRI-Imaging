from matplotlib import pyplot 
import numpy as np
import dicom
import dicom_numpy
import os

input_folder = './Sampdata/SE5/'
patients = os.listdir(input_folder)

patients.sort()
lstDCM = []


def load_scan(dir):
    for dirName,subdirList, fileList in os.walk(dir):
        for filename in fileList:
        	lstDCM.append(os.path.join(dirName,filename))
    return lstDCM 

first_patient = load_scan(input_folder)

refDS = dicom.read_file(lstDCM[0])

pixel_dim = (int(refDS.Rows),int(refDS.Columns),int(len(lstDCM)))

print(pixel_dim)

pixel_spacing = (float(refDS.PixelSpacing[0]),float(refDS.PixelSpacing[1]),float(refDS.SliceThickness))

print(pixel_spacing)

x = np.arange(0.0,(pixel_dim[0]+1)*(pixel_spacing[0]),pixel_spacing[0])
y = np.arange(0.0,(pixel_dim[1]+1)*(pixel_spacing[1]),pixel_spacing[1])
z = np.arange(0.0,(pixel_dim[2]+1)*(pixel_spacing[2]),pixel_spacing[2])

array_dicom = np.zeros(pixel_dim,dtype = refDS.pixel_array.dtype)


# In[22]:

for filename in lstDCM:
    da = dicom.read_file(filename)
    array_dicom[:,:,lstDCM.index(filename)] = da.pixel_array

#pyplot.figure(dpi = 1600)
#pyplot.axes().set_aspect('equal','datalim')
#pyplot.set_cmap(pyplot.gray())
#pyplot.pcolormesh(x,y,np.flipud(array_dicom[:,:,100]))
#pyplot.show()

array_dicom = np.flipud(array_dicom[:,:,:])

b = array_dicom[:,:,100].reshape(-1)
c = array_dicom.reshape(-1)

pyplot.hist(b,255,[0,4096])
pyplot.show()