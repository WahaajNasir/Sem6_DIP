import numpy as np
import cv2 as cv

def lower_by_x(image, thresh):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i][j] >= 0 and image[i][j] <= thresh):
                new_image[i][j] = 0
            elif (image[i][j] >= thresh+1 and image[i][j] <= 255):
                new_image[i][j] = 255

    return new_image

#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 6/Lab 6/lena.png", 0)
mean = np.mean(image)
median = np.median(image)

mean_thresh = lower_by_x(image, mean)
median_thresh = lower_by_x(image, median)

cv.imshow("Original Image", image)
cv.imshow("Mean Thresholding", mean_thresh)
cv.imshow("Median Thresholding", median_thresh)
cv.waitKey()