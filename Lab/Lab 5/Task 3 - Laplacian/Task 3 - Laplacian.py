import numpy as np
import cv2 as cv

def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*255

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

def remove_padding(padded_img, pad):
    rows, cols = padded_img.shape
    return padded_img[pad:rows-pad, pad:cols-pad]

def filter_laplace(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    laplace_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1 , -1]], dtype = int)
    laplace_img = np.zeros((rows, cols), dtype = np.float32)

    padded_img = padding(pad, image)
    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]
            value = np.sum(np.multiply(sub_img, laplace_filter))
            laplace_img[i][j] = value

    laplace_img = np.clip(image.astype(np.int16) + laplace_img, 0, 255)
    laplace_img = laplace_img.astype(np.uint8)

    return laplace_img


#---------------------
# Main
image =  cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 5/Lab 5/Fig03.tif", 0)

cv.imshow('Original Image', image)
cv.waitKey()

laplace = filter_laplace(image, 3)
cv.imshow('Laplace', laplace)
cv.waitKey()

