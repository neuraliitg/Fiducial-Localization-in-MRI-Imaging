import cv2
import numpy as np
import dicom
from matplotlib import pyplot as plt
import time
#import matplotlib
#from skimage.viewer import ImageViewer
dig = 1
while(1):
	
	kq = cv2.imread('fig1.png')
#cv2.imshow("fig",kq)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(25,25))
	kernel1 = np.ones((5,5),np.uint8)
	sample = dicom.read_file("./Sampdata/SE5/IM"+str(dig))
	image = sample.pixel_array
	print("DATATYPE : ",image.dtype)
	print("MAX PIXEL  VALUE :",image.max())

	norm_image = cv2.convertScaleAbs(image,alpha = (255.0/image.max()))

	print("Normalised Image Pixel",norm_image.max())

	cv2.imshow("Norm",norm_image)
	#plt.hist(norm_image.ravel(),256,[0,256])
	#plt.show()
	ret,thresh1 = cv2.threshold(norm_image,70,150,cv2.THRESH_BINARY)
	cv2.imshow("Thresholding",thresh1)




	#blur = cv2.bilateralFilter(norm_image,25,95,95)
	#kernel = np.ones((5,5),np.uint8)
	#erosion = cv2.erode(blur,kernel,iterations = 1)
	#cv2.imshow("Bilateral",blur)
	#edges = cv2.Canny(blur,180,200)
	#cv2.imshow("edges",edges)




	#dilation = cv2.dilate(blur,kernel,iterations = 1)
	#cv2.imshow("Dilation",dilation)
	#edgey = cv2.Canny(dilation,10,150)
	#cv2.imshow("Edges2",edgey)

	#tophat = cv2.morphologyEx(norm_image, cv2.MORPH_TOPHAT, kernel)
	#opening = cv2.morphologyEx(norm_image, cv2.MORPH_OPEN, kernel1)
#cv2.imshow("Morph",tophat)
#cv2.imshow("Morph_Open",opening)
	#canny = cv2.Canny(norm_image,180,200)
	#canny_morph = cv2.Canny(tophat,50,120)
#cv2.imshow("Canny",canny)
#cv2.imshow("Morph_Canny",canny_morph)
	_,contours, hierar = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
	rgbNorm = cv2.cvtColor(norm_image,cv2.COLOR_GRAY2RGB)
	print(len(contours))
	sz1 = len(contours)
	ar1 = np.arange(sz1).reshape((sz1,1))
	for i in range(0,sz1):
		ar1[i] = cv2.arcLength(contours[i],True)
	maxAr1 = np.amax(ar1)
	cindex1 = np.argmax(ar1)
	conFor = contours[cindex1]



	rgbNorm = cv2.drawContours(rgbNorm,contours,cindex1,(0,255,0),1)


	cv2.imshow("Contour",rgbNorm)
	dig = dig+1
	while(1):
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
#viewer = ImageViewer(norm_image)
#viewer.show()

cv2.destroyAllWindows()