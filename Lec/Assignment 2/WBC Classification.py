import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Histogram creation function
def histogram_creating(image):
    rows, cols = image.shape
    histogram = np.zeros(256, dtype=int)
    for i in range(rows):
        for j in range(cols):
            val = image[i][j]
            histogram[val] += 1
    return histogram

# PDF calculation
def calc_pdf(histogram, image):
    rows, cols = image.shape
    histogram = histogram / (rows * cols)
    return histogram

# Cumulative PDF calculation
def calc_cum_pdf(pdf):
    cum_pdf = np.zeros(len(pdf), dtype=float)
    cum_pdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cum_pdf[i] = cum_pdf[i - 1] + pdf[i]
    return cum_pdf

# Transformation function
def transformation_fun(cum_pdf):
    return np.uint8(cum_pdf * 255)

# Apply the transformation function to the image
def apply_trans(image, trans_fun):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            new_img[i][j] = trans_fun[image[i][j]]
    return new_img

# Function to compute Fourier transforms
def compute_fourier(image_channel):
    f = np.fft.fft2(image_channel)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum_norm = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX)
    return magnitude_spectrum_norm.astype(np.uint8)

# Main program
image = cv.imread(r"D:/Uni/Semester 6/DIP/Self/Lec/Assignment 2/wbc_data/Train/Eosinophil/Eosinophil_10.jpg", cv.IMREAD_COLOR)

# Split the image into individual channels
b, g, r = cv.split(image)

# Apply histogram equalization to each channel
b_histogram = histogram_creating(b)
g_histogram = histogram_creating(g)
r_histogram = histogram_creating(r)

b_pdf = calc_pdf(b_histogram, b)
g_pdf = calc_pdf(g_histogram, g)
r_pdf = calc_pdf(r_histogram, r)

b_cum_pdf = calc_cum_pdf(b_pdf)
g_cum_pdf = calc_cum_pdf(g_pdf)
r_cum_pdf = calc_cum_pdf(r_pdf)

transformation_function_b = transformation_fun(b_cum_pdf)
transformation_function_g = transformation_fun(g_cum_pdf)
transformation_function_r = transformation_fun(r_cum_pdf)

b_equalized = apply_trans(b, transformation_function_b)
g_equalized = apply_trans(g, transformation_function_g)
r_equalized = apply_trans(r, transformation_function_r)

# Compute Fourier transforms after histogram equalization (separate for each channel)
b_fft = compute_fourier(b_equalized)
g_fft = compute_fourier(g_equalized)
r_fft = compute_fourier(r_equalized)

# Show the original and equalized channels separately
cv.imshow("Original Blue Channel", b)
cv.imshow("Original Green Channel", g)
cv.imshow("Original Red Channel", r)

cv.imshow("Equalized Blue Channel", b_equalized)
cv.imshow("Equalized Green Channel", g_equalized)
cv.imshow("Equalized Red Channel", r_equalized)

# Show Fourier Transforms of the equalized channels
cv.imshow("Fourier - Blue Channel", b_fft)
cv.imshow("Fourier - Green Channel", g_fft)
cv.imshow("Fourier - Red Channel", r_fft)

# Compute and plot histograms for the equalized channels
hist_b = histogram_creating(b_equalized)
hist_g = histogram_creating(g_equalized)
hist_r = histogram_creating(r_equalized)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Histogram - Equalized Blue")
plt.plot(hist_b, color='blue')
plt.xlim([0, 255])

plt.subplot(1, 3, 2)
plt.title("Histogram - Equalized Green")
plt.plot(hist_g, color='green')
plt.xlim([0, 255])

plt.subplot(1, 3, 3)
plt.title("Histogram - Equalized Red")
plt.plot(hist_r, color='red')
plt.xlim([0, 255])

plt.tight_layout()
plt.show()

cv.waitKey()
cv.destroyAllWindows()
