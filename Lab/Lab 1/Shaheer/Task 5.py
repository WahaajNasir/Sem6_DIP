import numpy as np
import cv2 as cv

def draw_corner_boxes(image, padding):
    """Adds four colored boxes (black, blue, green, red) at the corners of the image."""
    modified_image = image.copy()
    height, width = image.shape[:2]

    # Define corner coordinates
    top_left = (0, 0, padding, padding)  # Black
    top_right = (0, width - padding, padding, width)  # Blue
    bottom_left = (height - padding, 0, height, padding)  # Green
    bottom_right = (height - padding, width - padding, height, width)  # Red

    # Apply colors to corners
    modified_image[top_left[0]:top_left[2], top_left[1]:top_left[3]] = [0, 0, 0]  # Black
    modified_image[top_right[0]:top_right[2], top_right[1]:top_right[3]] = [255, 0, 0]  # Blue
    modified_image[bottom_left[0]:bottom_left[2], bottom_left[1]:bottom_left[3]] = [0, 255, 0]  # Green
    modified_image[bottom_right[0]:bottom_right[2], bottom_right[1]:bottom_right[3]] = [0, 0, 255]  # Red

    return modified_image

# User inputs
height = int(input("Enter image height: "))
width = int(input("Enter image width: "))
padding = int(input("Enter box padding size: "))

# Create white image
base_img = np.full((height, width, 3), 255, dtype=np.uint8)

# Add colored boxes
modified_img = draw_corner_boxes(base_img, padding)

# Display images
cv.imshow('Original Image', base_img)
cv.imshow('Modified Image', modified_img)

cv.waitKey(0)
cv.destroyAllWindows()
