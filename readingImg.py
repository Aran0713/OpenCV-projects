# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:50:15 2022

@author: Aran A
"""

import cv2 as cv

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)
  



img = cv.imread('Photos/cat.jpg')
rescale_img = rescaleFrame(img)
#cv.imshow('cat', img)
#cv.imshow('cat2', rescale_img)

img2 = cv.imread('Photos/dog.jpg')
rescale_img2 = rescaleFrame(img2)
cv.imshow('dog', img2)
cv.imshow('dog2', rescale_img2)
  
cv.waitKey(0)

