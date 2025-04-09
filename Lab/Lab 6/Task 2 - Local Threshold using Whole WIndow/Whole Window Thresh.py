import cv2 as cv
import numpy as np

def local_thresh_mean(image, filter_size):
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(0, rows, filter_size):
        for j in range(0, cols, filter_size):
            sub_img = image[i:i+filter_size, j:j+filter_size]
            mean_val = np.mean(sub_img)
            for a in range(sub_img.shape[0]):
                for b in range(sub_img.shape[1]):
                    if(sub_img[a][b] <= mean_val):
                        filtered_img[i+a][j+b] = 0
                    else:
                        filtered_img[i+a][j+b] = 255

    return filtered_img

def local_thresh_median(image, filter_size):
    rows, cols = image.shape
    filtered_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(0, rows, filter_size):
        for j in range(0, cols, filter_size):
            sub_img = image[i:i+filter_size, j:j+filter_size]
            median_val = np.median(sub_img)
            for a in range(sub_img.shape[0]):
                for b in range(sub_img.shape[1]):
                    if(sub_img[a][b] <= median_val):
                        filtered_img[i+a][j+b] = 0
                    else:
                        filtered_img[i+a][j+b] = 255

    return filtered_img

#Main
image = cv.imread(r"D:\Uni\Semester 6\DIP\Self\Lab\Lab Mid\Open Lab with Instructions\dataset\images\render0359.png", 0)
mean_img = local_thresh_mean(image, 3)
median_img = local_thresh_median(image, 3)

cv.imshow("Original Image", image)
cv.imshow("Mean Thresholding", mean_img)
cv.imshow("Median Thresholding", median_img)
cv.waitKey()