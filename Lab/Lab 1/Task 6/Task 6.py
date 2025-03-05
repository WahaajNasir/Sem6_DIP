import numpy as np
import cv2 as cv

def vertical_mirror_flip(image):
    flipped_img = image.copy()
    rows, cols = image.shape

    mid_row = rows // 2

    # If odd, we take one extra row for bottom half to match the image size
    if rows % 2 == 1:
        flipped_img[mid_row + 1:, :] = np.flipud(image[:mid_row, :])
    else:
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
