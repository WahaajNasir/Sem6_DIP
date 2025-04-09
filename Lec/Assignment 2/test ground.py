import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread(r"D:\Uni\Semester 6\DIP\Self\Lec\Assignment 2\wbc_data\Train\Basophil\Basophil_5.jpg")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define HSV range for purple (WBC nucleus and cytoplasm)
lower_purple = np.array([120, 40, 40])
upper_purple = np.array([170, 255, 255])
mask = cv.inRange(image_hsv, lower_purple, upper_purple)

# Morphological operations
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_OPEN, kernel, iterations=1)

# Apply mask to original image
white_background = np.ones_like(image_rgb, dtype=np.uint8) * 255
isolated = np.where(mask_clean[:, :, np.newaxis] == 255, image_rgb, white_background)

# Show output
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)

plt.subplot(1, 3, 2)
plt.title("HSV Mask (Purple)")
plt.imshow(mask_clean, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Final WBC Isolated")
plt.imshow(isolated)

plt.tight_layout()
plt.show()
