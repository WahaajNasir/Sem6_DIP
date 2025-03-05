import numpy as np
import cv2 as cv

def compute_euclidean_distance(size):
    center_x, center_y = size // 2, size // 2
    distance_map = np.zeros((size, size), dtype=np.float32)

    for y in range(size):
        for x in range(size):
            distance_map[y, x] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    return distance_map

def normalize_and_invert(image):
    max_val = np.max(image)
    normalized = (image / max_val) * 255
    return normalized.astype(np.uint8)

def quantize_image(image, levels):
    step = 256 // levels
    return (image // step) * step

size = 501

white_image = np.full((size, size), 255, dtype=np.uint8)

distance_map = compute_euclidean_distance(size)

normalized_map = normalize_and_invert(distance_map)

distance_map_16 = quantize_image(normalized_map, 16)
distance_map_4 = quantize_image(normalized_map, 4)
distance_map_1 = (normalized_map > 128).astype(np.uint8) * 255  # Binary

cv.imshow("Euclidean Distance Map (Smooth 255 to 0)", normalized_map)
cv.imshow("16 Levels", distance_map_16)
cv.imshow("4 Levels", distance_map_4)
cv.imshow("1 Level (Binary)", distance_map_1)

cv.waitKey(0)
cv.destroyAllWindows()
