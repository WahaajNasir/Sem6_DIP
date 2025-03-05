import numpy as np
import cv2 as cv

def apply_padding(height, width, padding_size, image):
    padded_image = np.full((height + 2 * padding_size, width + 2 * padding_size), 255, dtype=np.uint8)
    padded_image[padding_size:height + padding_size, padding_size:width + padding_size] = image
    return padded_image

height = int(input("Enter the number of rows: "))
width = int(input("Enter the number of columns: "))
padding_size = int(input("Enter padding size: "))

original_img = np.zeros((height, width), dtype=np.uint8)
padded_img = apply_padding(height, width, padding_size, original_img)

cv.imshow("Original Image", original_img)
cv.waitKey()

cv.imshow("Padded Image", padded_img)
cv.waitKey()
cv.destroyAllWindows()
