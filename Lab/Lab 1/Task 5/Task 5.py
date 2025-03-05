import numpy as np
import cv2 as cv

def add_boxes(image, pad):
    edit_image = orig_image.copy()
    rows, cols = image.shape[:2]
    #Black
    edit_image[0:pad, 0:pad] = [0, 0 , 0]

    #Blue
    edit_image[0:pad,cols-pad:cols] = [255, 0 ,0]

    #Green
    edit_image[rows-pad:rows, 0:pad] = [0, 255, 0]

    #Red
    edit_image[rows-pad:rows, cols-pad:cols] = [0, 0 , 255]

    return edit_image
rows = int(input("Enter rows: "))
cols = int(input("Enter columns: "))
pad = int(input("Enter box padding: "))

orig_image = np.ones((rows, cols, 3), dtype = np.uint8)*255
box_img = add_boxes(orig_image, pad)

cv.imshow('Window1', orig_image)
cv.waitKey()

cv.imshow('Window2', box_img)
cv.waitKey()
