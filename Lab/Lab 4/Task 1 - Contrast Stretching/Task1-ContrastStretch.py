import cv2 as cv
import numpy as np

def contrast_stretch(image):
    im_min_5 = np.percentile(image, 5)
    im_max_95 = np.percentile(image, 95)
    rows,cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if(image[i][j] < im_min_5):
                new_img[i][j] = 0
            elif(image[i][j] > im_max_95):
                new_img[i][j] = 255
            else:
                new_img[i][j] = 255 * ((image[i][j] - im_min_5) / (im_max_95 - im_min_5))

    return new_img


#---------------------------------
# Main

image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 4/Lab 4/low_con.jpg", 0)
cont_img = contrast_stretch(image)

cv.imshow('Original Image', image)
cv.imshow('Contrast Image', cont_img)
cv.waitKey()

