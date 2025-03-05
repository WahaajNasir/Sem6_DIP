import numpy as np
import cv2 as cv

def pattern1_edit(height, width, spacing, image):
    modified_image = np.copy(image)
    for row in range(height):
        for col in range(0, width, spacing * 2):
            modified_image[row, col:col + spacing] = 0  # Black strip
    return modified_image

def pattern2_edit(height, width, size, image):
    modified_image = np.zeros((height, width), dtype=np.uint8)  # Start with black image
    center_row, center_col = height // 2, width // 2
    modified_image[center_row - size:center_row + size, center_col - size:center_col + size] = 255  # White square
    return modified_image

def pattern3_edit(height, width, spacing, image):
    modified_image = np.copy(image)
    box_size = spacing * 3  # Bigger white box

    # Horizontal black lines
    for row in range(box_size, height, box_size + spacing):
        modified_image[row:row + spacing, :] = 0  # Black row

    # Vertical black lines
    for col in range(box_size, width, box_size + spacing):
        modified_image[:, col:col + spacing] = 0  # Black column

    return modified_image

# Input from user
height = int(input("Enter the number of rows: "))
width = int(input("Enter the number of columns: "))
spacing = int(input("Enter spacing size: "))

# Creating a white image
base_image = np.full((height, width), 255, dtype=np.uint8)

# Generating modified images
edited_img1 = pattern1_edit(height, width, spacing, base_image)
edited_img2 = pattern2_edit(height, width, spacing, base_image)
edited_img3 = pattern3_edit(height, width, spacing, base_image)

# Display images
cv.imshow('Original Image', base_image)

cv.imshow('Pattern 1', edited_img1)

cv.imshow('Pattern 2', edited_img2)

cv.imshow('Pattern 3', edited_img3)

cv.waitKey(0)
cv.destroyAllWindows()
