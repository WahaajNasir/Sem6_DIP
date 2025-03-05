import numpy as np
import cv2 as cv

def reduce_resolution(img, factor=4):
    """Downsamples the image by taking every nth pixel."""
    return img[::factor, ::factor]

# Load and resize the image to 512x512
image_path = "D:/Uni/Semester 6/DIP/Self/Lab/Lab 1/lab1.png"
original_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
original_img = cv.resize(original_img, (512, 512), interpolation=cv.INTER_LINEAR)

# Downsample using NumPy slicing
downsampled_img = reduce_resolution(original_img)

# Save and display images
cv.imwrite("D:/Uni/Semester 6/DIP/Self/Lab/Lab 1/lab1_downsample.png", downsampled_img)

cv.imshow("Original Image", original_img)
cv.imshow("Downsampled Image", downsampled_img)

cv.waitKey(0)
cv.destroyAllWindows()
