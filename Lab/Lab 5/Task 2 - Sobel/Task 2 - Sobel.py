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

def filter_sobel(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    mag_img = np.zeros((rows, cols), dtype = np.float32)
    phase_img = np.zeros((rows, cols), dtype=np.float32)

    filter_x = np.zeros((3, 3), dtype=int)
    filter_y = np.zeros((3, 3), dtype=int)

    # Filter_X
    filter_x[0][0] = -1
    filter_x[0][1] = -2
    filter_x[0][2] = -1
    filter_x[1][0:3] = 0
    filter_x[2][0] = 1
    filter_x[2][1] = 2
    filter_x[2][2] = 1

    # Filter_Y
    filter_y[0][0] = -1
    filter_y[0:3][1] = 0
    filter_y[0][2] = 1
    filter_y[1][0] = -2
    filter_y[1][2] = 2
    filter_y[2][0] = -1
    filter_y[2][2] = 1

    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sobel_x = np.sum(np.multiply(sub_img, filter_x))
            sobel_y = np.sum(np.multiply(sub_img, filter_y))
            mag = np.sqrt(sobel_x**2 + sobel_y**2)
            phase = np.arctan2(sobel_y, sobel_x)
            mag_img[i][j] = mag
            phase_img[i][j] = phase

    mag_max = np.max(mag_img)
    mag_min = np.min(mag_img)
    mag_img = ((mag_img - mag_min)/(mag_max - mag_min)) *255
    phase_img = ((phase_img - np.min(phase_img))/(np.max(phase_img) - np.min(phase_img))) *255

    mag_img = mag_img.astype(np.uint8)
    phase_img = phase_img.astype(np.uint8)

    return mag_img, phase_img


#---------------------
# Main
image =  cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 5/Lab 5/Fig03.tif", 0)

cv.imshow('Original Image', image)
cv.waitKey()

mag_img, phase_img = filter_sobel(image, 3)

cv.imshow('Magnitude Image', mag_img)
cv.imshow('Phase Image', phase_img)
cv.waitKey()