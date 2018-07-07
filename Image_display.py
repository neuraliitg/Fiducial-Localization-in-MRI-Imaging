
# coding: utf-8

# In[2]:

import numpy as np
import dicom
import dicom_numpy
from matplotlib import pyplot
from matplotlib import pylab
import os
import cv2

# In[3]:

df = dicom.read_file("./Datasets/MyHead/MR000082.dcm")


# In[4]:

df


# In[5]:

pylab.imshow(df.pixel_array,cmap = pylab.cm.bone)
pylab.savefig('fig1.png')

# In[6]:

pylab.show()


# In[7]:
#kq = cv2.imread('fig.png')
#cv2.imshow("fig",kq)

input_folder = './Datasets/MyHead/'


# In[8]:

patients = os.listdir(input_folder)


# In[9]:

patients.sort()


# In[10]:

lstDCM = []


# In[11]:

def load_scan(dir):
    for dirName,subdirList, fileList in os.walk(dir):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstDCM.append(os.path.join(dirName,filename))
    return lstDCM       


# In[12]:

first_patient = load_scan(input_folder)


# In[13]:

#print(first_patient)


# In[14]:

refDS = dicom.read_file(lstDCM[0])


# In[15]:

pixel_dim = (int(refDS.Rows),int(refDS.Columns),int(len(lstDCM)))


# In[16]:

print(pixel_dim)


# In[17]:

pixel_spacing = (float(refDS.PixelSpacing[0]),float(refDS.PixelSpacing[1]),float(refDS.SliceThickness))


# In[18]:

print(pixel_spacing)


# In[19]:

x = np.arange(0.0,(pixel_dim[0]+1)*(pixel_spacing[0]),pixel_spacing[0])
y = np.arange(0.0,(pixel_dim[1]+1)*(pixel_spacing[1]),pixel_spacing[1])
z = np.arange(0.0,(pixel_dim[2]+1)*(pixel_spacing[2]),pixel_spacing[2])


# In[20]:

#print(x)


# In[21]:

array_dicom = np.zeros(pixel_dim,dtype = refDS.pixel_array.dtype)


# In[22]:

for filename in lstDCM:
    da = dicom.read_file(filename)
    array_dicom[:,:,lstDCM.index(filename)] = da.pixel_array


# In[ ]:




# In[25]:

pyplot.figure(dpi = 1600)
pyplot.axes().set_aspect('equal','datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x,y,np.flipud(array_dicom[:,:,160]))


# In[26]:

#pyplot.show()


# In[27]:




# In[31]:

operations = np.flipud(array_dicom[:,:,160])


# In[32]:

print(operations)

#cv2.imshow("fig",operations)
#cv2.waitKey()
#cv2.destroyAllWindows()
# In[35]:

#image1 = pylab.imshow(df.pixel_array,cmap = pylab.cm.bone)


# In[36]:

#image1


# In[37]:

#print(image1)


# In[44]:




# In[ ]:




# In[ ]:



