from pickletools import uint8

import numpy as np
import cv2 as cv

def lower_by_2(image):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i, j] >= 0 and image[i, j] <= 127):
                new_image[i, j] = 0
            elif (image[i, j] >= 128 and image[i, j] <= 255):
                new_image[i, j] = 255

    return new_image

# s = (L-1)-r
def neg_img(image):
    l = 256
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)
    for i in range(rows):
        for j in range(cols):
            r = int(image[i][j])
            s = (256-1)-r
            new_img[i][j] = np.uint8(s)

    return new_img

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 3/Lab 3/gradient.png", 0)
image_bin = lower_by_2(image)

thresh = np.mean(image)
thresh_bin = np.mean(image_bin)
print(f"Threshold value: {thresh}")
print(f"Threshold value of bin img: {thresh_bin}")

image_neg = neg_img(image)

cv.imshow('Original Image', image)
cv.imshow('Negative Image', image_neg)
cv.imshow('2 Levels', image_bin)
cv.waitKey()