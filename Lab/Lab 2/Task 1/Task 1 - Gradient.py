import numpy as np
import cv2 as cv

def lower_by_16(image):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if(image[i, j] >= 0 and image[i, j] <= 15):
                new_image[i, j] = 0
            elif(image[i, j] >= 16 and image[i, j] <= 31):
                new_image[i, j] = 15
            elif(image[i, j] >= 32 and image[i, j] <= 47):
                new_image[i, j] = 31
            elif(image[i, j] >= 48 and image[i, j] <= 63):
                new_image[i, j] = 47
            elif(image[i, j] >= 64 and image[i, j] <= 79):
                new_image[i, j] = 63
            elif(image[i, j] >= 80 and image[i, j] <= 95):
                new_image[i, j] = 79
            elif(image[i, j] >= 96 and image[i, j] <= 111):
                new_image[i, j] = 95
            elif(image[i, j] >= 112 and image[i, j] <= 127):
                new_image[i, j] = 111
            elif(image[i, j] >= 128 and image[i, j] <= 143):
                new_image[i, j] = 127
            elif(image[i, j] >= 144 and image[i, j] <= 159):
                new_image[i, j] = 143
            elif(image[i, j] >= 160 and image[i, j] <= 175):
                new_image[i, j] = 159
            elif(image[i, j] >= 176 and image[i, j] <= 191):
                new_image[i, j] = 175
            elif(image[i, j] >= 192 and image[i, j] <= 207):
                new_image[i, j] = 191
            elif(image[i, j] >= 208 and image[i, j] <= 223):
                new_image[i, j] = 207
            elif(image[i, j] >= 224 and image[i, j] <= 239):
                new_image[i, j] = 223
            elif(image[i, j] >= 240 and image[i, j] <= 255):
                new_image[i, j] = 239
    return new_image

def lower_by_4(image):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if(image[i, j] >= 0 and image[i, j] <= 63):
                new_image[i, j] = 0
            elif(image[i, j] >= 64 and image[i, j] <= 127):
                new_image[i, j] = 63
            elif(image[i, j] >= 128 and image[i, j] <= 191):
                new_image[i, j] = 127
            elif(image[i, j] >= 192 and image[i, j] <= 255):
                new_image[i, j] = 191

    return new_image

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

img_path = "D:/Uni/Semester 6/DIP/Self/Lab/Lab 2/gradient.png"
orig_img = cv.imread(img_path, 0)

down_16 = lower_by_16(orig_img)
down_4 = lower_by_4(orig_img)
down_2 = lower_by_2(orig_img)

cv.imshow('Window1', orig_img)
cv.waitKey()

cv.imshow('Window1', down_16)
cv.waitKey()

cv.imshow('Window1', down_4)
cv.waitKey()

cv.imshow('Window1', down_2)
cv.waitKey()