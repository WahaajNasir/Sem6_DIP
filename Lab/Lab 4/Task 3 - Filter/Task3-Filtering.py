import cv2 as cv
import numpy as np

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

def filter(image, filter_size, filter_var):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, filtered_img)
    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sub_img = np.multiply(sub_img, filter_var)
            sub_img_val = np.sum(sub_img)

            padded_img[i][j] = sub_img_val

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

# ----------------------------------
# Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 4/Lab 4/fig05.tif", 0)
cv.imshow('Original Image', image)
cv.waitKey()

filtered = filter(image, 3, 1/9)
cv.imshow('Filtered Image', filtered)
cv.waitKey()

