import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread(r"D:\Uni\Semester 6\DIP\Self\Lab\Lab 5\Lab 5\Fig01.tif", cv2.IMREAD_GRAYSCALE)

# Define Sobel filters
sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Your Sobel-X
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Standard Sobel-Y

# Apply filters using OpenCV
sobel_x_result = cv2.filter2D(image, cv2.CV_64F, sobel_x)  # Horizontal edges
sobel_y_result = cv2.filter2D(image, cv2.CV_64F, sobel_y)  # Vertical edges

# Compute magnitude of gradients
sobel_magnitude = np.sqrt(sobel_x_result**2 + sobel_y_result**2)
sobel_magnitude = np.uint8(sobel_magnitude)

# Plot the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(sobel_x_result, cmap="gray")
plt.title("Sobel-X (Horizontal Edges)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(sobel_y_result, cmap="gray")
plt.title("Sobel-Y (Vertical Edges)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(sobel_magnitude, cmap="gray")
plt.title("Edge Magnitude")
plt.axis("off")

plt.show()
