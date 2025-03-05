import numpy as np
import cv2 as cv

def img_mean_transforms(image):
    rows, cols = image.shape
    mean = int(np.mean(image))
    print(mean)
    img1 = np.zeros((rows, cols), dtype = np.uint8)
    img2 = np.zeros((rows, cols), dtype=np.uint8)
    img3 = np.zeros((rows, cols), dtype=np.uint8)

    #Cond 1
    for i in range(rows):
        for j in range(cols):
            val = int(image[i][j])
            if((val >= 0) & (val <= mean)):
                img1[i][j] = 0
            elif((val > mean) & (val <= 255)):
                img1[i][j] = 255

    #Cond 2
    for i in range(rows):
        for j in range(cols):
            val = int(image[i][j])
            if ((val >= 0) & (val <= mean)):
                img2[i][j] = 255
            elif ((val > mean) & (val <= 255)):
                img2[i][j] = 0

    #Cond 3
    for i in range(rows):
        for j in range(cols):
            val = int(image[i][j])
            if((val >= mean-20) & (val <= mean+20)):
                img3[i][j] = 0
            else:
                img3[i][j] = 255

    cv.imshow('Original Image', image)
    cv.imshow('Cond 1', img1)
    cv.imshow('Cond 2', img2)
    cv.imshow('Cond 3', img3)
    cv.waitKey()

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 3/Lab 3/gradient.png", 0)
img_mean_transforms(image)