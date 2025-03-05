import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*0

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

def lower_by_x(image, thresh):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i, j] >= 0 and image[i, j] <= thresh):
                new_image[i, j] = 0
            elif (image[i, j] >= thresh+1 and image[i, j] <= 255):
                new_image[i, j] = 255

    return new_image

def lower_by_2(image):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i, j] >= 0 and image[i, j] <= 127):
                new_image[i, j] = 0
            elif (image[i, j] >= 128 and image[i, j] <= 255):
                new_image[i, j] = 255

    return new_image

def cc(orig):
    rows, cols = orig.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8)
    my_dict = {}
    count = 1

    for i in range(1, rows):
        for j in range(1, cols):
            if orig[i][j] == 255:
                neighbors = []  # Store nonzero neighboring labels

                # Check all 8-connected neighbors
                if orig[i - 1][j] == 255:
                    neighbors.append(new_img[i - 1][j])
                if orig[i][j - 1] == 255:
                    neighbors.append(new_img[i][j - 1])
                if orig[i - 1][j - 1] == 255:
                    neighbors.append(new_img[i - 1][j - 1])
                if j + 1 < cols and orig[i - 1][j + 1] == 255:
                    neighbors.append(new_img[i - 1][j + 1])

                if not neighbors:  # No connected neighbors, assign new label
                    new_img[i][j] = count
                    my_dict[count] = count
                    count += 1
                else:
                    min_label = min(neighbors)
                    new_img[i][j] = min_label

                    # Merge equivalence classes
                    for label in neighbors:
                        root1 = find_root(my_dict, min_label)
                        root2 = find_root(my_dict, label)
                        if root1 != root2:
                            my_dict[max(root1, root2)] = min(root1, root2)

    for i in range(1, rows):
        for j in range(1, cols):
            if new_img[i][j] != 0:
                new_img[i][j] = find_root(my_dict, new_img[i][j])

    return new_img, my_dict


# Path compression to find root label
def find_root(my_dict, x):
    if x not in my_dict:
        my_dict[x] = x
        return x
    while my_dict[x] != x:
        my_dict[x] = my_dict[my_dict[x]]  # Path compression
        x = my_dict[x]
    return x

def histogram_creating(image):
    rows, cols = image.shape
    histogram = np.zeros(256, dtype = int)

    for i in range(rows):
        for j in range(cols):
            val = image[i][j]
            histogram[val] += 1

    return histogram

def hist_cumsum(histogram):
    cumsum = np.zeros(len(histogram), dtype = int)
    cumsum[0] = histogram[0]
    for i in range(1, len(histogram)):
        cumsum[i] = cumsum[i-1] + histogram[i]

    return cumsum

#Main
total_threshold = 0

for i in range (3, 241):
    if i < 10:
        temp = "00" + str(i)
    elif i < 100:
        temp = "0" + str(i)
    else:
        temp = str(i)

    image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/" + temp + ".bmp",0)  # Grayscale image
    test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/masks/" + temp + ".png", 0)

    true_mask = lower_by_2(test_img)

    histogram = histogram_creating(image)

    cumsum = hist_cumsum(histogram)
    cdf = cumsum/max(cumsum)

    #Extracting first value where the value is 30%
    threshold = (np.where(cdf >= 0.3)[0][0])
    print(f"For img {str(i)} threshold value is: {threshold}")
    total_threshold = total_threshold + threshold

avg_threshold = total_threshold//238

print(f"Avg Thresh: {avg_threshold}")
temp =""
for i in range (3, 241):
    if i < 10:
        temp = "00" + str(i)
    elif i < 100:
        temp = "0" + str(i)
    else:
        temp = str(i)

    image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/" + temp + ".bmp",0)  # Grayscale image
    test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/masks/" + temp + ".png", 0)

    true_mask = lower_by_2(test_img)

    segmented = np.bitwise_not(lower_by_x(image, avg_threshold))
    #Check visually how the borders look
    seg_check = np.bitwise_not(lower_by_x(image, avg_threshold))

    #Turns it into a binary array
    segmented = (segmented > avg_threshold).astype(np.uint8)
    true_mask = (true_mask > 127).astype(np.uint8)

    true_positives = np.sum((segmented == 1) & (true_mask == 1))
    false_positives = np.sum((segmented == 1) & (true_mask == 0))

    actual_wbc = np.sum(true_mask == 1)

    normal_true = true_positives/actual_wbc
    normal_false = false_positives/actual_wbc

    print(f"True Positives = {true_positives}")
    print(f"False Positives = {false_positives}")
    print(f"Normal True = {normal_true}")
    print(f"Normal False = {normal_false}")
    print("\n\n")

print("\n\n\n-----------------------\nNow Using TEST\n-----------------------------")
total_tp = 0
total_fp = 0
total_actual_wbc = 0
for i in range (241, 301):
    if i < 10:
        temp = "00" + str(i)
    elif i < 100:
        temp = "0" + str(i)
    else:
        temp = str(i)

    image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/images/" + temp + ".bmp",0)  # Grayscale image
    test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/masks/" + temp + ".png", 0)

    segmented = np.bitwise_not(lower_by_x(image, avg_threshold))
    seg_check = np.bitwise_not(lower_by_x(image, avg_threshold))

    # Turns it into a binary array
    segmented = (segmented > avg_threshold).astype(np.uint8)
    true_mask = (test_img > 127).astype(np.uint8)

    true_positives = np.sum((segmented == 1) & (true_mask == 1))
    false_positives = np.sum((segmented == 1) & (true_mask == 0))

    actual_wbc = np.sum(true_mask == 1)

    normal_true = true_positives / actual_wbc
    normal_false = false_positives / actual_wbc

    print(f"True Positives = {true_positives}")
    print(f"False Positives = {false_positives}")
    print(f"Normal True = {normal_true}")
    print(f"Normal False = {normal_false}")
    print("\n\n")

    total_tp += true_positives
    total_fp += false_positives
    total_actual_wbc += actual_wbc

if total_actual_wbc > 0:  # Avoid division by zero
    total_tp_percentage = (total_tp / total_actual_wbc) * 100
    total_fp_percentage = (total_fp / total_actual_wbc) * 100
    if((total_tp + total_fp) > 0):
        overall_accuracy = (total_tp / (total_tp + total_fp)) * 100
    else:
        overall_accuracy = 0
else:
    total_tp_percentage = total_fp_percentage = overall_accuracy = 0

print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total Actual WBC Pixels: {total_actual_wbc}")
print(f"\nTrue Positive Percentage: {total_tp_percentage:.2f}%")
print(f"False Positive Percentage: {total_fp_percentage:.2f}%")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

#cv.imshow('Window 1', image)
#cv.imshow('Window 2', seg_check)
#print(image.shape)
#cv.waitKey()
