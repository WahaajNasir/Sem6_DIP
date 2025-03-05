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

def filter(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    padded_img = padding(pad, image)
    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sub_img_val = np.sum(sub_img)/(filter_size * filter_size)

            padded_img[i][j] = sub_img_val

    filtered_img = remove_padding(padded_img, pad)

    return filtered_img

def histogram_creating(image):
    rows, cols = image.shape
    histogram = np.zeros(256, dtype = int)

    for i in range(rows):
        for j in range(cols):
            val = image[i][j]
            histogram[val] += 1

    return histogram

def calc_pdf(histogram, image):
    rows, cols = image.shape

    histogram = histogram/(rows*cols)

    return histogram

def calc_cum_pdf(pdf):
    cum_pdf = np.zeros(len(pdf), dtype = float)
    cum_pdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cum_pdf[i] = cum_pdf[i-1] + pdf[i]

    return cum_pdf

def transformation_fun(cum_pdf):
    return np.uint8((cum_pdf*255))

def apply_trans(image, trans_fun):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            new_img[i][j] = trans_fun[image[i][j]]

    return new_img

# ----------------------------------
# Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 4/Lab 4/landscape.jpeg", 0)
cv.imshow('Original Image', image)
cv.waitKey()

filtered = filter(image, 3)

histogram = histogram_creating(filtered)
pdf = calc_pdf(histogram, filtered)
cum_pdf = calc_cum_pdf(pdf)
transformation_function = transformation_fun(cum_pdf)

equilized_img = apply_trans(filtered, transformation_function)
print(image.shape)
print(equilized_img.shape)
cv.imshow('Enhanced Image', equilized_img)
cv.waitKey()
