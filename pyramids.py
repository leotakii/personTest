# import the necessary packages

#from pyimagesearch.helpers import pyramid
#from skimage.transform import pyramid_gaussian

import argparse
import cv2
import re as regex
 
#"atalho/train_64x128_H96/pos.lst"
#"atalho/train_64x128_H96/neg.lst"
def readFile(filePath):
    images = []

    #print filePath
    try:
        fp = open(filePath,"r")
        line = fp.readline()
        cnt = 1

        tempPath = filePath


        while line:
            imagePath = line.strip().split("/")
            tempArrayPath = tempPath.strip().split("/")

            if regex.search("^*train_64x128_H96*$",filePath): #erro na regex!!!!!
                print filePath
                tempPatempArrayPath.remove("pos.lst")
                tempArrayPath.remove("neg.lst")
                imagePath.remove("train")
                '/'.join(a)

            imagePath = filePath + imagePath
            #print(imagePath)
            image = cv2.imread(imagePath)
            print(image)
            images.append(image)
            line = fp.readline()
            cnt += 1

    finally:
        fp.close()
        


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="", help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
 
# load the image
filePath = "atalho/train_64x128_H96/pos.lst"
images = readFile(filePath)

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

