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

def local_thresh_mean(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, filtered_img)
    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            mean_val = np.mean(sub_img)
            if(image[i][j] <= mean_val):
                padded_img[i][j] = 0
            else:
                padded_img[i][j] = 255

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

def local_thresh_median(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, filtered_img)
    for i in range(pad,rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            median_val = np.median(sub_img)
            if(image[i][j] <= median_val):
                padded_img[i][j] = 0
            else:
                padded_img[i][j] = 255

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 6/Lab 6/lena.png", 0)
mean_img = local_thresh_mean(image, 3)
median_img = local_thresh_median(image, 3)

cv.imshow("Original Image", image)
cv.imshow("Mean Thresholding", mean_img)
cv.imshow("Median Thresholding", median_img)
cv.waitKey()