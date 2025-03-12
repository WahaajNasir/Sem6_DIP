import cv2 as cv
import numpy as np

def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.zeros((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

def remove_padding(padded_img, pad):
    rows, cols = padded_img.shape
    return padded_img[pad:rows-pad, pad:cols-pad]

# See page 169 of Gonzalez 4th Ed for formula
def gaussian_filter(size, sigma=1, k=1):
    box = np.zeros((size, size), dtype=np.float32)
    center = size//2
    sum_val = 0

    for i in range(size):
        for j in range(size):
            x = i-center
            y = j-center
            box[i][j] = k* (np.exp(-(x**2+y**2)/(2*sigma**2)))
            sum_val += box[i][j]

    return box/sum_val  #Normalized so that sum is 1

def apply_gaussian_filter(image, size=5, sigma=1):
    kernel = gaussian_filter(size, sigma)  # Call Gaussian filter inside
    pad = size // 2
    padded_image = padding(pad, image)
    rows, cols = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+size, j:j+size]
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image.astype(np.uint8)

def filter_sobel(image, filter_size):
    pad = filter_size//2
    rows, cols = image.shape
    mag_img = np.zeros((rows, cols), dtype = np.float32)
    phase_img = np.zeros((rows, cols), dtype=np.float32)

    filter_x = np.zeros((3, 3), dtype=int)
    filter_y = np.zeros((3, 3), dtype=int)

    # Filter_X
    filter_x[0][0] = -1
    filter_x[0][1] = -2
    filter_x[0][2] = -1
    filter_x[1][0:3] = 0
    filter_x[2][0] = 1
    filter_x[2][1] = 2
    filter_x[2][2] = 1

    # Filter_Y
    filter_y[0][0] = -1
    filter_y[0:3][1] = 0
    filter_y[0][2] = 1
    filter_y[1][0] = -2
    filter_y[1][2] = 2
    filter_y[2][0] = -1
    filter_y[2][2] = 1

    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            sub_img = image[i-pad:i+pad+1, j-pad:j+pad+1]
            sobel_x = np.sum(np.multiply(sub_img, filter_x))
            sobel_y = np.sum(np.multiply(sub_img, filter_y))
            mag = np.sqrt(sobel_x**2 + sobel_y**2)
            phase = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
            mag_img[i][j] = mag
            phase_img[i][j] = phase

    mag_max = np.max(mag_img)
    mag_min = np.min(mag_img)
    mag_img = ((mag_img - mag_min)/(mag_max - mag_min)) *255

    mag_img = mag_img.astype(np.uint8)

    return mag_img, phase_img


def quantize_angles(phase_img):
    quantized = np.zeros_like(phase_img, dtype=np.uint8)
    angles = [0, 45, 90, 135]

    for i in range(phase_img.shape[0]):
        for j in range(phase_img.shape[1]):
            angle = phase_img[i, j] % 180  # Normalize to [0, 180]
            min_diff = float('inf')
            closest_angle = 0

            for a in angles:
                diff = abs(a - angle)
                if diff < min_diff:
                    min_diff = diff
                    closest_angle = a

            quantized[i][j] = closest_angle

    return quantized

def non_maxima_suppression(mag_img, phase_img):
    pad = 1

    rows, cols = mag_img.shape
    suppressed = np.zeros((rows, cols), dtype =np.uint8)

    padded_mag = padding(1, mag_img)
    padded_phase = padding(1, phase_img)

    for i in range(rows):
        for j in range(cols):
            phase = padded_phase[i+pad][j+pad]
            mag = padded_mag[i+pad][j+pad]

            #Check Up and Down
            if phase == 0:
                n1 = padded_mag[i+pad-1][j+pad]
                n2 = padded_mag[i+pad+1][j+pad]
            elif phase == 45:   #Diagonal right_bottom and top_left
                n1 = padded_mag[i+pad+1][j+pad+1]
                n2 = padded_mag[i+pad-1][j+pad-1]
            elif phase == 90:   #Left and Right
                n1 = padded_mag[i+pad][j+pad+1]
                n2 = padded_mag[i+pad][j+pad-1]
            elif phase == 135:  #Diagonal left_bottom and top_right
                n1 = padded_mag[i+pad+1][j+pad-1]
                n2 = padded_mag[i+pad-1][j+pad+1]

            if mag >= n1 and mag >= n2:
                suppressed[i][j] = mag
            else:
                suppressed[i][j] = 0

    return suppressed

def hysterisis_thresholding(image):
    rows, cols = image.shape
    max_val = np.max(image)
    high_thresh = 0.75 * max_val
    low_thresh = 0.25 * max_val
    print(high_thresh)
    print(low_thresh)
    print(np.unique(image))
    pad = 1
    threshold_img = padding(1, image)
    for i in range(threshold_img.shape[0]):
        for j in range(threshold_img.shape[1]):
            if threshold_img[i][j] >= high_thresh:
                threshold_img[i][j] = 1
            elif threshold_img[i][j] <= low_thresh:
                threshold_img[i][j] = 0
            else:
                threshold_img[i][j] = 0.5

    print(np.unique(threshold_img))
    for i in range(rows):
        for j in range(cols):
            if threshold_img[i][j] == 0.5:
                box = threshold_img[i+pad-1:i+pad+2, j+pad-1:j+pad+2]
                if np.any(box==1):
                    threshold_img[i+pad][j+pad] = 1
                else:
                    threshold_img[i+pad][j+pad] = 0

    threshold_img = remove_padding(threshold_img, 1)
    return threshold_img


def canny_edge_detection(image):
    gaussian_blur = apply_gaussian_filter(image, 5, 1)
    mag_img, phase_img = filter_sobel(image, 3)
    phase_img = quantize_angles(phase_img)
    suppressed_image = non_maxima_suppression(mag_img, phase_img)
    threshold_img = hysterisis_thresholding(suppressed_image)*255

    return threshold_img


#Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 5/Lab 5/Fig01.tif", 0)
edge_image = canny_edge_detection(image)

cv.imshow("Original", image)
cv.imshow("Edge Detection Image", edge_image)
cv.waitKey()
