import numpy as np
import cv2 as cv

def downsample(image):
    downsample_arr = np.ones((128, 128), dtype = np.uint8)

    for i in range(0, 512, 4):
        for j in range(0,512,4):
            downsample_arr[int(i/4)][int(j/4)] = image[i][j]
    return downsample_arr

image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 1/lab1.png", 0)

image = cv.resize(image, (512, 512))
image_down = downsample(image)

cv.imwrite("D:/Uni/Semester 6/DIP/Self/Lab/Lab 1/lab1_downsample.png", image_down)
cv.imshow('Window1', image)
cv.waitKey(5)

cv.imshow('Window2', image_down)
cv.waitKey()