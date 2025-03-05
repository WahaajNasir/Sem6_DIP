import numpy as np
import cv2 as cv

def img1_edit(rows, cols, pad, orig):
    edit_img = orig.copy()
    for i in range(rows):
        for j in range(0, cols, pad*2):
            for k in range(pad):
                if j + k < cols:
                    edit_img[i][j+k] = 0
    return edit_img

def img2_edit(rows, cols, pad ,orig):
    edit_img = orig.copy() * 0
    mid_row = int(rows/2)
    mid_col = int(cols/2)

    top_left_r = mid_row - pad
    top_left_c = mid_col - pad
    bottom_right_r = mid_row + pad
    bottom_right_c = mid_col + pad

    edit_img[top_left_r:bottom_right_r, top_left_c:bottom_right_c] = 255

    return edit_img

def img3_edit(rows, cols, pad, orig):
    edit_img = orig.copy()

    box_size = pad * 3  # White box should be bigger than the black lines

    # Draw horizontal black lines
    for i in range(box_size, rows, box_size + pad):
        for k in range(pad):  # Black line thickness
            if i + k < rows:
                edit_img[i + k, :] = 0  # Set entire row to black

    # Draw vertical black lines
    for j in range(box_size, cols, box_size + pad):
        for k in range(pad):
            if j + k < cols:
                edit_img[:, j + k] = 0  # Set entire column to black

    return edit_img

rows = int(input("Enter no of rows: "))
cols = int(input("Enter no of columns: "))
pad = int(input("Enter padding for boxes: "))
orig_img = np.ones((rows, cols), dtype = np.uint8)*255 #This creates white box thingy

orig_img_edit_1 = img1_edit(rows, cols, pad, orig_img)
orig_img_edit_2 = img2_edit(rows, cols, pad, orig_img)
orig_img_edit_3 = img3_edit(rows, cols, pad, orig_img)

cv.imshow('Window1',orig_img)
cv.waitKey()

cv.imshow('Window1',orig_img_edit_1)
cv.waitKey()

cv.imshow('Window1',orig_img_edit_2)
cv.waitKey()

cv.imshow('Window1',orig_img_edit_3)
cv.waitKey()