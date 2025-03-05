import numpy as np
import cv2 as cv

def padding(rows, cols, pad, orig):
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*255

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

rows = int(input("Enter no of rows: "))
cols = int(input("Enter no of cols: "))
pad = int(input("Enter padding: "))

img = np.zeros((rows, cols), dtype = np.uint8)
pad_img = padding(rows, cols, pad, img)
cv.imshow('Window1', img)
cv.waitKey()
cv.imshow('Window1', pad_img)
cv.waitKey()