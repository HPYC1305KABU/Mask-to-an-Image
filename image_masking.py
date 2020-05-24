#!/usr/bin/env python
 
'''
Welcome to the Image Masking! 
 
Usage:
  image_masking.py [<image>]
 
Keys:
  r     - mask the image
  SPACE - reset the inpainting mask
  ESC   - exit
'''

from __future__ import print_function
 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import sys 
from common import Sketcher
 
INPUT_IMAGE = "image01.png"
IMAGE_NAME = INPUT_IMAGE[:INPUT_IMAGE.index(".")]
MASK_IMAGE = IMAGE_NAME + "_mask.png"
TABLE_IMAGE = IMAGE_NAME + "_table.jpg"
 
def main():

    # Pull system arguments
    try:
        fn = sys.argv[1]
    except:
        fn = INPUT_IMAGE
 
    # Load the image and store into a variable
    image = cv2.imread(cv2.samples.findFile(fn))
 
    if image is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
 
    # Create an image for sketching the mask
    image_mark = image.copy()
    sketch = Sketcher('Image', [image_mark], lambda : ((255, 255, 255), 255))
 
    # Sketch a mask
    while True:
        ch = cv2.waitKey()
        if ch == 27: # ESC - exit
            break
        if ch == ord('r'): # r - mask the image
            break
        if ch == ord(' '): # SPACE - reset the inpainting mask
            image_mark[:] = image
            sketch.show()
 
    # define range of white color in HSV
    lower_white = np.array([0,0,255])
    upper_white = np.array([255,255,255])
 
    # Create the mask
    mask = cv2.inRange(image_mark, lower_white, upper_white)
 
    # Create the inverted mask
    mask_inv = cv2.bitwise_not(mask)
 
    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Extract the dimensions of the original image
    rows, cols, channels = image.shape
    image = image[0:rows, 0:cols]
 
    # Bitwise-OR mask and original image
    colored_portion = cv2.bitwise_or(image, image, mask = mask)
    colored_portion = colored_portion[0:rows, 0:cols]
 
    # Bitwise-OR inverse mask and grayscale image
    gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
    gray_portion = np.stack((gray_portion,)*3, axis=-1)
 
    # Combine the two images
    output = colored_portion + gray_portion
    
    # save the mask 
    cv2.imwrite(MASK_IMAGE, mask)
    
    # Create a table showing input image, mask, and output
    mask = np.stack((mask,)*3, axis=-1)
    table_of_images = np.concatenate((image, mask, output), axis=1)
    cv2.imwrite(TABLE_IMAGE, table_of_images)
 
    # Display images, used for debugging
    #cv2.imshow('Original Image', image)
    #cv2.imshow('Sketched Mask', image_mark)
    cv2.imshow('Mask', mask)
    #cv2.imshow('Output Image', output)
    cv2.imshow('Table of Images', table_of_images)
    cv2.waitKey(0) # Wait for a keyboard event
 
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
