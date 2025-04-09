import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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


# ---------------------------------
# Main
image = cv.imread(r"D:\Uni\Semester 6\DIP\Self\Lab\Lab Mid\Open Lab with Instructions\dataset\images\render0359.png", 0)
cv.imshow('Original Image', image)
cv.waitKey()
histogram = histogram_creating(image)

plt.plot(histogram)
plt.xlabel('Pixel Indices')
plt.ylabel('Pixel Count')
plt.title('Histogram')
plt.show()

pdf = calc_pdf(histogram, image)
plt.plot(pdf)
plt.xlabel('Pixel Indices')
plt.ylabel('Pixel Intensity')
plt.title('Probability Density Function')
plt.show()

cum_pdf = calc_cum_pdf(pdf)
plt.plot(cum_pdf)
plt.xlabel('Pixel Indices')
plt.ylabel('Pixel Intensity')
plt.title('Cumulative PDF')
plt.show()

transformation_function = transformation_fun(cum_pdf)
plt.plot(transformation_function)
plt.xlabel('Pixel Indices')
plt.ylabel('Pixel Count')
plt.title('Transformation Function')
plt.show()

print(transformation_function)

equilized_img = apply_trans(image, transformation_function)
cv.imshow("Equilized Image", equilized_img)
cv.waitKey()
