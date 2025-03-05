import numpy as np
import cv2 as cv

def grayslicing(image):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            val = int(image[i][j])
            if((val >= 100) and (val <= 200)):
                new_img[i][j] = 210
            else:
                new_img[i][j] = image[i][j]

    return new_img

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 3/Lab 3/gradient.png", 0)
gray_img = grayslicing(image)

cv.imshow('Original', image)
cv.imshow('Graysliced', gray_img)
cv.waitKey()