import skimage
import numpy as np
import dicom
import os
import dicom_numpy
from skimage import morphology as morph
from skimage import restoration
from skimage.filters import threshold_otsu
from scipy import ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import math
from skimage import feature
##########################################################################################################

#function to load slices locations inside the array lstDCM...
def load_scan(dir):
    for dirName,subdirList, fileList in os.walk(dir):
        for filename in fileList:
            lstDCM.append(os.path.join(dirName,filename))
    return lstDCM 

#function to display DICOM slice using SimpleITK...
def sitk_show(img, title = None, margin = 0.05, dpi = 40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin)*nda.shape[0] / dpi, (1+ margin)* nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0],0)
    fig = plt.figure(figsize = figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1-2*margin, 1-2*margin])
    
    plt.set_cmap("gray")
    ax.imshow(nda,extent = extent, interpolation = None)
    
    if title:
        plt.title(title)
    
    plt.show()

#function to find the truncated array for performing convolution...(Part of optimization)
def straight(image):
    fixX = int((refDS.Rows)/2)
    fixY = int((refDS.Columns)/2)
    fixZ = int(len(lstDCM)/2)
    i = 0
    j = 0
    k = 0
    l = 0
    m = 0
    n = 0
    for i in range(fixY):
        conv = image[fixX:fixX+5,i:i+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t1 = [fixX,i,fixZ]
            print(t1)
            break
    for j in range(fixX):
        conv = image[j:j+5,fixY:fixY+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t2 = [j,fixY,fixZ]
            print(t2)
            break 
    for k in range(int(refDS.Rows),1,-1):
        conv = image[k-5:k,fixY:fixY+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t3 = [k,fixY,fixZ]
            print(t3)
            break 
    for l in range(int(refDS.Columns),1,-1):
        conv = image[fixX:fixX+5,l-5:l,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t4 = [fixX,l,fixZ]
            print(t4)
            break 
    for m in range(fixZ):
        conv = image[fixX:fixX+5,fixY:fixY+5,m:m+5]
        s = np.sum(conv)
        if s>0:
            t5 = [fixX,fixY,m]
            print(t5)
            break
    for n in range(int(len(lstDCM)),0,-1):
        conv = image[fixX:fixX+5,fixY:fixY+5,n-5:n]
        s = np.sum(conv)
        if s>0:
            t6 = [fixX,fixY,n]
            print(t6)
            break        
    return [i,j,k,l,m,n]        

#Raindeer optimization[Deciding Fitting the sphere inside the skull volume to reduce iterations...]
def adjust_sphere(image,centroid):
    x_dim = int(image.shape[0])
    y_dim = int(image.shape[1])
    z_dim = int(image.shape[2])
    xi = int(centroid[0])
    yi = int(centroid[1])
    zi = int(centroid[2])
    image[xi-10:xi+10,yi-10:yi+10,zi-10:zi+10]
    arg_x = np.zeros((x_dim),dtype=int)
    arg_y = np.zeros((y_dim),dtype=int)
    arg_z = np.zeros((z_dim),dtype=int)
    for i in range(xi,5,-1):                        #For reverse X-direction....from centroid to left octant
        conv = image[i-5:i,yi:yi+5,zi:zi+5]
        s = np.sum(conv)
        arg_x[i] = s
        if s>0 and i>11:
            for j in range(i,10,-1):
                conv1 = image[j-10:j,yi:yi+10,zi:zi+10]
                s1 = np.sum(conv1)
                arg_x[j] = s1
    for i in range(xi,x_dim-5+1):                        #For forward X-direction....from centroid to right octant
        conv = image[i:i+5,yi:yi+5,zi:zi+5]
        s = np.sum(conv)
        arg_x[i] = s
        if s>0 and i>11:
            for j in range(i,x_dim-10+1):
                conv1 = image[j:j+10,yi:yi+10,zi:zi+10]
                s1 = np.sum(conv1)
                arg_x[j] = s1
                
    for i in range(yi,5,-1):                        #For reverse Y-direction....from centroid to left octant
        conv = image[xi:xi+5,i-5:i,zi:zi+5]
        s = np.sum(conv)
        arg_y[i] = s
        if s>0 and i>11:
            for j in range(i,10,-1):
                conv1 = image[xi:xi+10,j-10:j,zi:zi+10]
                s1 = np.sum(conv1)
                arg_y[j] = s1
    for i in range(yi,y_dim-5+1):                        #For forward Y-direction....from centroid to right octant
        conv = image[xi:xi+5,i:i+5,zi:zi+5]
        s = np.sum(conv)
        arg_y[i] = s
        if s>0 and i>11:
            for j in range(i,y_dim-10+1):
                conv1 = image[xi:xi+10,j:j+10,zi:zi+10]
                s1 = np.sum(conv1)
                arg_y[j] = s1

    for i in range(zi,5,-1):                        #For reverse Z-direction....from centroid to left octant
        conv = image[xi:xi+5,yi:yi+5,i-5:i]
        s = np.sum(conv)
        arg_z[i] = s
        if s>0 and i>11:
            for j in range(i,10,-1):
                conv1 = image[xi:xi+10,yi:yi+10,j-10:j]
                s1 = np.sum(conv1)
                arg_z[j] = s1
    for i in range(zi,z_dim-5+1):                        #For forward Z-direction....from centroid to right octant
        conv = image[xi:xi+5,yi:yi+5,i:i+5]
        s = np.sum(conv)
        arg_z[i] = s
        if s>0 and i>11:
            for j in range(i,z_dim-10+1):
                conv1 = image[xi:xi+10,yi:yi+10,j:j+10]
                s1 = np.sum(conv1)
                arg_z[j] = s1
                
    x_neg = np.argmax(arg_x[0:xi])
    x_pos = np.argmax(np.append(np.zeros(xi),arg_x[xi:x_dim]))
    y_neg = np.argmax(arg_y[0:yi])
    y_pos = np.argmax(np.append(np.zeros(yi),arg_y[yi:y_dim]))
    z_neg = np.argmax(arg_z[0:zi])
    z_pos = np.argmax(np.append(np.zeros(zi),arg_z[zi:z_dim]))
    print(arg_z)
    return [[x_neg,x_pos],[y_neg,y_pos],[z_neg,z_pos]]

#Adding the ball inside the skull..
def add_ball(image1,radius,center):
    image = image1.copy()
    ball = morph.ball(radius)
    #print(image.shape)
    x = int(center[0])
    y = int(center[1])
    z = int(center[2])
    image[x-radius:x+radius+1,y-radius:y+radius+1,z-radius:z+radius+1] = -5000*(image[x-radius:x+radius+1,y-radius:y+radius+1,z-radius:z+radius+1] + ball)
    return image

#3D kernel convolution inside the binary array..
def nd_convolution_mod(image2,kernel):
    image = image2.copy()
    print(image.shape)
    r = image.shape
    convolution = np.zeros([(refDS.Rows - 49 + 1),(refDS.Columns - 49 +1),(len(lstDCM) - 49 +1)],dtype = int)
    rowRange = r[0] - 49 + 1
    colRange = r[1] - 49 +1
    depRange = r[2] - 49 +1
    for i in range(0,rowRange):
        for j in range(0,colRange):
            for k in range(0,depRange):
                conv = np.multiply(image[i:i+49,j:j+49,k:k+49],kernel)
                print(conv.shape)
                s = np.sum(conv)
                if s<0:
                    break
                if s>0:
                    convolution[i+33][j+100][k+6] = s
            for l in range(r[2]-1,k+49,-1):
                conv1 = np.multiply(image[i:i+48+1,j:j+48+1,l-49:l],kernel)
                print(conv1.shape)
                s = np.sum(conv1)
                if s<0:
                    break
                if s>0:
                    convolution[i][j][l-49] = s
        
    return convolution

#Finding the maxima of the neighbourhood..
ddef center_shift2(convolute,tup,factor,iter_max):
    vote_max = 0
    vote_min = 0
    
    if len(tup) == 0:
        return
    value = convolute[tup[0],tup[1],tup[2]]
    center = tup.copy()
    dim = 5
    window = convolute[center[0]-dim:center[0]+dim,center[1]-dim:center[1]+dim,center[2]-dim:center[2]+dim]
    vote_min = len(np.asarray(np.where(window < value)).T)
    vote_max = len(np.asarray(np.where(window > value)).T)
    face_val = len(np.asarray(np.where(window == value)).T)
    print([vote_min,vote_max,face_val])
    if (vote_min/((dim*2)**3)) >= factor:
        return center
        
    if (vote_min/((dim*2)**3)) < factor:
        for i in range(iter_max):
            max_val = window.max()
            print(max_val)
            index = np.asarray(np.where(convolute == max_val)).T
            dist = np.zeros(len(index),dtype = int)
            for i in range(0,len(index)):
                dist[i] = sqrt(((center[0]-index[i][0])**2) + ((center[1]-index[i][1])**2) + ((center[2]-index[i][2])**2))
            k = np.argmin(dist)
        
            center = index[k]
            window = convolute[center[0]-dim:center[0]+dim,center[1]-dim:center[1]+dim,center[2]-dim:center[2]+dim]
            vote_min = len(np.asarray(np.where(window < max_val)).T)
            vote_max = len(np.asarray(np.where(window > max_val)).T)
            face_val = len(np.asarray(np.where(window == max_val)).T)
            if (vote_min/((dim*2)**3)) >= factor:
                return center     
                break

def straight_mod(image):
    fixX = int((refDS.Rows)/2)
    fixY = int((refDS.Columns)/2)
    fixZ = int(len(lstDCM)/2)
    i = 0
    j = 0
    k = 0
    l = 0
    m = 0
    n = 0
    for i in range(fixY):
        conv = image[fixX:fixX+5,i:i+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t1 = [fixX,i,fixZ]
            print(t1)
            break
    for j in range(fixX):
        conv = image[j:j+5,fixY:fixY+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t2 = [j,fixY,fixZ]
            print(t2)
            break 
    for k in range(int(refDS.Rows),1,-1):
        conv = image[k-5:k,fixY:fixY+5,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t3 = [k,fixY,fixZ]
            print(t3)
            break 
    for l in range(int(refDS.Columns),1,-1):
        conv = image[fixX:fixX+5,l-5:l,fixZ:fixZ+5]
        s = np.sum(conv)
        if s>0:
            t4 = [fixX,l,fixZ]
            print(t4)
            break 
    for m in range(fixZ):
        conv = image[fixX:fixX+5,fixY:fixY+5,m:m+5]
        s = np.sum(conv)
        if s>0:
            t5 = [fixX,fixY,m]
            print(t5)
            break
    for n in range(int(len(lstDCM)),0,-1):
        conv = image[fixX:fixX+5,fixY:fixY+5,n-5:n]
        s = np.sum(conv)
        if s>0:
            t6 = [fixX,fixY,n]
            print(t6)
            break        
    return [t1,t2,t3,t4,t5,t6]        

###########################################################################################################

#sorting file names inside an array.....
input_folder = './Sampdata/SE5/'
patients = os.listdir(input_folder)
patients.sort()
lstDCM = []

#Defining array spacing and dimensions....
first_patient = load_scan(input_folder)
refDS = dicom.read_file(lstDCM[0])
pixel_dim = (int(refDS.Rows),int(refDS.Columns),int(len(lstDCM)))
print(pixel_dim)
pixel_spacing = (float(refDS.PixelSpacing[0]),float(refDS.PixelSpacing[1]),float(refDS.SliceThickness))
print(pixel_spacing)
x = np.arange(0.0,(pixel_dim[0]+1)*(pixel_spacing[0]),pixel_spacing[0])
y = np.arange(0.0,(pixel_dim[1]+1)*(pixel_spacing[1]),pixel_spacing[1])
z = np.arange(0.0,(pixel_dim[2]+1)*(pixel_spacing[2]),pixel_spacing[2])

#preparing 3D array of the scan...
array_dicom = np.zeros(pixel_dim,dtype = refDS.pixel_array.dtype)
for filename in lstDCM:
    da = dicom.read_file(filename)
    array_dicom[:,:,(int(da.InstanceNumber)-1)] = da.pixel_array

array_dicom.shape

#Yash's 2D function goes here...











#Thresholding for getting fiducial points + some noise .....
result = (array_dicom < 1600) * array_dicom
result = (result > 1100) * result

#Otsu thresholding for getting the binary image..

thres_min = threshold_otsu(result)
boolean_binary = (result > thres_min)
binary = 1 * boolean_binary

#Trimmin the binary array for convolution..
st = straight(binary)
trimmed_binary = binary[st[1]-10:st[2]+10,st[0]-10:st[3]+10,st[4]-2:st[5]+2]
trimmed_binary.shape

#Finding centroid of the skull..
props = measure.regionprops(label_image=trimmed_binary,cache=True)
centroid = props[0].centroid
print("Centroid : "centroid)

r_2 = props[0].equivalent_diameter
print("Equivalent Diameter: ",r_2)

exp = trimmed_binary.copy()    #Image copy
#Spherical kernel for convolution...
r = 50                  
k = morph.ball(r)
k.shape

#Parameters to fit a sphere inside the binary skull..
adjust_param = adjust_sphere(trimmed_binary,centroid)
dist = [[(centroid[0]-adjust_param[0][0]),(adjust_param[0][1]-centroid[0])],[(centroid[1]-adjust_param[1][0]),(adjust_param[1][1]-centroid[1])],[(centroid[2]-adjust_param[2][0]),(adjust_param[2][1]-centroid[2])]]
km = np.ravel(dist)
km.sort()
km
#Deciding the radius of the sphere...
ki = int(km[3])
ri = int((ki-1)/2)
ball_1 = morph.ball(ri)

fit_model = add_ball(trimmed_binary,ri,centroid)
ker = morph.ball(24)
ker.shape
#Costliest step...
den_res = nd_convolution_mod(fit_model,ker)
cut = den_res.copy()
cut[281-60:281+60,241-160:241+160,44-30:44+60] = 0   #excluded global maxima
#Finding all local maximas in the density map..
peaks4 = feature.peak_local_max(cut,min_distance=10,threshold_abs=785,indices=True,num_peaks=20)
peaks4
#Seeds selection script goes here...




#Defining labels for segmentation of fiducials.......
labelFid = 1
labelBack = 2

#Using SimpleITK GDCM reader....
reader = sitk.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(input_folder)
reader.SetFileNames(filenamesDICOM)
imgOriginal  =reader.Execute()      #3D array with object specifications

img = imgOriginal[:,:,21]
#sitk_show(img)

#Curvature Flow algorithm for image smoothening...
imgSmooth = sitk.CurvatureFlow(image1 = imgOriginal,timeStep = 0.125, numberOfIterations= 5)

#Fiducial seeds..
seeds = [(335,155,21),(330,146,22)]

#Copying image into new array...
seeds_im = sitk.Image(imgSmooth)

for s in seeds:
    seeds_im[s] = 10000

#Confidence connected Filter for image segmentation(Region growing algorithm..)
imgSeg = sitk.ConfidenceConnected(image1 = imgSmooth,seedList = [(330,146,22),(325,145,21)],numberOfIterations=7,multiplier=1.0, replaceValue=labelFid)

#Binary Hole filling for improving segmentation area
imgSeg_NoHoles = sitk.VotingBinaryHoleFilling(image1 = imgSeg,radius = [1]*8,majorityThreshold=1,backgroundValue=0,foregroundValue = labelFid)

#Gets statistical feature of image region...
stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(imgSeg,imgSmooth)

#Centroid label of the fiducial region...
m = stats.GetCentroid(stats.GetLabels()[0])

#Physical Index to voxel coordinate transform..
r = sitk.Image.TransformPhysicalPointToContinuousIndex(imgSmooth,m)

#Patient coordinate transform goes here,....







