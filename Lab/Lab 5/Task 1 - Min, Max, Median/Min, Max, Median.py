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

def filter_min(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, filtered_img)
    for i in range(pad, rows+pad):
        for j in range(pad, cols+pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sub_img_val = np.min(sub_img)
            padded_img[i][j] = sub_img_val

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

def filter_max(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, image)
    for i in range(pad, rows+pad):
        for j in range(pad, cols+pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sub_img_val = np.max(sub_img)
            padded_img[i][j] = sub_img_val

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

def filter_median(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, filtered_img)
    for i in range(pad, rows+pad):
        for j in range(pad, cols+pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sub_img_val = np.median(sub_img)
            padded_img[i][j] = sub_img_val

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

#-----------------
# Main

image =  cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 5/Lab 5/Fig01.tif", 0)

max_img_3 = filter_max(image, 3)
min_img_3 = filter_min(image, 3)
median_img_3 = filter_median(image, 3)

max_img_15 = filter_max(image, 15)
min_img_15 = filter_min(image, 15)
median_img_15 = filter_median(image, 15)

max_img_31 = filter_max(image, 31)
min_img_31 = filter_min(image, 31)
median_img_31 = filter_median(image, 31)


cv.imshow('Original', image)
cv.imshow('Max Filter', max_img_3)
cv.imshow('Min Filter', min_img_3)
cv.imshow('Median Filter', median_img_3)
cv.waitKey()
cv.destroyAllWindows()

cv.imshow('Original', image)
cv.imshow('Max Filter', max_img_15)
cv.imshow('Min Filter', min_img_15)
cv.imshow('Median Filter', median_img_15)
cv.waitKey()
cv.destroyAllWindows()

cv.imshow('Original', image)
cv.imshow('Max Filter', max_img_31)
cv.imshow('Min Filter', min_img_31)
cv.imshow('Median Filter', median_img_31)
cv.waitKey()
cv.destroyAllWindows()
