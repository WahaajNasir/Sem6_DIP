import numpy as np
import cv2 as cv

def vertical_mirror_flip(image):
    """Flips the top half of the image and mirrors it onto the bottom half."""
    flipped_img = image.copy()
    mid_row = image.shape[0] // 2  # Find middle row

    # Flip the top half and copy it to the bottom half
    flipped_img[mid_row:, :] = np.flipud(image[:mid_row, :])

    return flipped_img

# Load grayscale image
image_path = "D:/Uni/Semester 6/DIP/Self/Lab/Lab 1/lab1.png"
original_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Display original image
cv.imshow("Original Image", original_img)
cv.waitKey()

# Apply vertical mirror flip
flipped_img = vertical_mirror_flip(original_img)

# Display flipped image
cv.imshow("Flipped Image", flipped_img)
cv.waitKey(0)
cv.destroyAllWindows()
