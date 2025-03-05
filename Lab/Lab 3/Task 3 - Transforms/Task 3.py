import numpy as np
import cv2 as cv
import math

def transform_power(image):
    rows, cols = image.shape
    img1 = np.zeros((rows, cols), dtype=np.uint8)
    img2 = np.zeros((rows, cols), dtype=np.uint8)
    img3 = np.zeros((rows, cols), dtype=np.uint8)
    img4 = np.zeros((rows, cols), dtype=np.uint8)

    #Gamma = 0.2
    for i in range(rows):
        for j in range(cols):
            r = image[i][j]
            s = int(255 * ((r/255)**0.2))
            img1[i][j] = s

    #Gamma = 0.5
    for i in range(rows):
        for j in range(cols):
            r = image[i][j]
            s = int(255 * ((r/255)**0.5))
            img2[i][j] = s

    #Gamma = 1.2
    for i in range(rows):
        for j in range(cols):
            r = image[i][j]
            s = int(255 * ((r/255)**1.2))
            img3[i][j] = s

    #Gamma = 1.8
    for i in range(rows):
        for j in range(cols):
            r = image[i][j]
            s = int(255 * ((r/255)**1.8))
            img4[i][j] = s

    cv.imshow('Original', image)
    cv.imshow('Gamma 0.2', img1)
    cv.imshow('Gamma 0.5', img2)
    cv.imshow('Gamma 1.2', img3)
    cv.imshow('Gamma 1.8', img4)
    cv.waitKey()

def log_transform(image):
    c = 255 /np.log10(1+int(np.max(image))) #1+255 in np.max returns 0 as it wraps around
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            r = int(image[i][j])
            s = c * (np.log10(r+1))
            new_img[i][j] = np.uint8(s)

    return new_img

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 3/Lab 3/fig02.tif", 0)
transform_power(image)
log_img = log_transform(image)

print(np.max(image))
cv.imshow('Log Transform', log_img)
cv.waitKey()