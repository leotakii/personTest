# import the necessary packages

#from pyimagesearch.helpers import pyramid
#from skimage.transform import pyramid_gaussian

import argparse
import cv2
import re as regex
import os
import numpy as np
from PIL import Image, ImageDraw
import sys
import matplotlib.pyplot as plt
import math
 
#"atalho/train_64x128_H96/pos.lst"
#"atalho/train_64x128_H96/neg.lst"

def drange(start, stop, step):
	while start < stop:
			yield start
			start += step

def getAllImages(realpath):
    path = realpath

    image_paths = [os.path.join(path, f) for f in os.listdir(path) if  f.endswith('.png')]
    # labels will contains the label that is assigned to the image
    images = []
    labels = [] #obter labels !!!

    #print(image_paths)
    #print(os.listdir(path))
    print("Loading database: "+ path)
    for image_path in image_paths:	#LOOP PARA OBTER IMAGENS .png
        #print(image_path)
        A = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        A = np.float32(A) / 255.0
        #A = np.fromfile(image_path, dtype='int8', sep="")
        #print(A.shape)
        #A = A.reshape([28793, -1])
        #A = Image.fromarray(A)
        images.append(A)


    return (images,labels)        

def readDatabase(path):
    if path == '70X134H96':
        path = 'atalho/' + path + '/Test/pos/'

    elif path == '96X160H96':
        path = 'atalho/' + path + '/Train/pos/'

    elif path == 'train_64x128_H96':
        path = 'atalho/' + path + '/pos/'  
    
    else:
        print "ERRO NA LEITURA: Bases disponiveis: \'96X160H96\', \'70X134H96\', \'train_64x128_H96\'"
        return 1

    return getAllImages(path)


def getPyramids(image, levels=3):
    """
    image : image matrix
    levels : quantity of levels in the pyramid
    returns [imageOriginal, imageLevel1, imageLevel2, ...]
    """
    pyr = []
    pyr.append(image)

    for level in range(levels):
        image = cv2.pyrDown(image)
        pyr.append(image)

    return pyr

def histogramHOG(mag, angle,cellSize=8): #cellsize = 8
	allHistograms = []
	bin = []
	for index_x in drange(0,mag.shape[0],cellSize): #CALCULA MEDIA E STDEV ENTRE OS PIXELS DE UM BLOCO
		for index_y in drange(0,mag.shape[1],cellSize):
			bin = [0.0]*9 #0, 20, 40, 60 ... 160 (9 bins)
			for i in range(cellSize):
				for j in range(cellSize):
					angleModified = angle[index_x+i][index_y+j] % 180  #este angulo precisa estar entre 0 e 180
					firstIndex = int(math.floor(angleModified/20 - 0.5) % 9) #calcula os bins alvo
					secondIndex =  int(math.ceil(angleModified/20 - 0.5) % 9)
					proportion =  abs(angleModified/20 -0.5 - int (angleModified/20 -0.5)) #proporcao
					invProportion = 1 - proportion
					bin[firstIndex]  += proportion * mag[index_x+i][index_y+j]
					bin[secondIndex] += invProportion * mag[index_x+i][index_y+j]
			for k in range(0,9):
				print bin[k],
			print
			print len(bin)
			allHistograms.append(bin)
	return allHistograms
			

def normalizeHistogram(cellByCellHistogramList,sizeHori,sizeVert):
	normalizedHist = []
	sizeHori/=8
	sizeVert/=8
	count = 0
	print (sizeHori,sizeVert)
	for i in range (0,sizeHori-1): 
		for j in range (0,sizeVert-1):
			normalizedHist = cellByCellHistogramList[i+j] + cellByCellHistogramList[i+j+1] + cellByCellHistogramList[i+j+sizeHori] + cellByCellHistogramList[i+j+sizeHori+1]
			#print(i+j,i+j+1)
			#print(i+j+sizeHori,i+j+sizeHori+1)
			histogramNorm = np.linalg.norm(normalizedHist) + 0.000001 #epsilon para nao dividir por 0
			normalizedHist /= histogramNorm
			count +=1
	cellSize = 8
	print count
	
	

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="", help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
 
# load the image
#filePath = "70X134H96"
#filePath = "96X160H96"
filePath = "train_64x128_H96"
images,labels = readDatabase(filePath)

for image in images:

	getPyramids(image)
	gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	#cv2.imshow("Image",mag)
	#cv2.waitKey(1000)
	cellByCellHistogramList = histogramHOG(mag, angle,8)
	normalizedFourCellsHistogram = normalizeHistogram(cellByCellHistogramList,mag.shape[0],mag.shape[1])
	exit(0)


#print(images[0])
# METHOD #1: No smooth, just scaling.
# loop over the image pyramid

'''
for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
	# show the resized image
	cv2.imshow("Layer {}".format(i + 1), resized)
	cv2.waitKey(0)
 
# close all windows
cv2.destroyAllWindows()
'''
 
'''
# METHOD #2: Resizing + Gaussian smoothing.
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
	# if the image is too small, break from the loop
	if resized.shape[0] < 30 or resized.shape[1] < 30:
		break
		
	# show the resized image
	cv2.imshow("Layer {}".format(i + 1), resized)
	cv2.waitKey(0)
'''

