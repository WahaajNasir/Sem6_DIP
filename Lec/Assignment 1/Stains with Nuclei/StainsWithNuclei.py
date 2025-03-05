import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*255

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

def remove_padding(padded_img, pad):
    rows, cols = padded_img.shape
    return padded_img[pad:rows-pad, pad:cols-pad]

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
            if new_img[i][j] > 0:
                new_img[i][j] = find_root(my_dict, new_img[i][j])

    return new_img, my_dict


# Path compression to find root label
def find_root(my_dict, x):
    #Added to avoid that the background coming in the dictionaries
    if x == 0:
        return 0
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

def cyto_to_gray(cyto_img, cyto_dict):
    rows, cols = cyto_img.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if cyto_img[i][j] in cyto_dict:
                new_img[i][j] = 128

    return new_img

def nuclei_to_white(nucleus_img, nucleus_dict):
    rows, cols = nucleus_img.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if nucleus_img[i][j] in nucleus_dict:
                new_img[i][j] = 255

    return new_img

def merge_for_mask(cyto_img, nuclei_img):
    rows, cols = cyto_img.shape
    mask = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if((cyto_img[i][j] == 128) & (nuclei_img[i][j] != 255)):
                mask[i][j] = 128
            elif((cyto_img[i][j] == 128) & (nuclei_img[i][j] == 255)):
                mask[i][j] = 255
            elif((cyto_img[i][j] != 128) & (nuclei_img[i][j] == 255)):
                mask[i][j] = 0 #Reduce false positives of nucleus as nuclei should only be inside cyto

    return mask

#D.C = 2 * (X ∩ Y) / X  + Y
#X is Predicted Pixels
#Y is Actual Pixels
#X ∩ Y is true Positives
def calculate_dice_coefficient(true_mask, own_mask, label):
    rows, cols = true_mask.shape
    X = 0
    Y = 0
    TP = 0

    for i in range(rows):
        for j in range(cols):
            if(own_mask[i][j] == label):
                X += 1

    for i in range(rows):
        for j in range(cols):
            if (true_mask[i][j] == label):
                Y += 1

    for i in range(rows):
        for j in range(cols):
            if ((own_mask[i][j] == label) & (true_mask[i][j] == label)):
                TP += 1

    DC = (2 * TP) / (X+Y)

    return DC

#Purely for checking purposes
def neg_img(image):
    l = 256
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)
    for i in range(rows):
        for j in range(cols):
            r = int(image[i][j])
            s = (256-1)-r
            new_img[i][j] = np.uint8(s)

    return new_img

#Main
total_threshold_cyto = 0
total_threshold_nucleus = 0

for i in range (3, 241):
    if i < 10:
        temp = "00" + str(i)
    elif i < 100:
        temp = "0" + str(i)
    else:
        temp = str(i)

    image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/" + temp + ".bmp",0)  # Grayscale image
    test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/masks/" + temp + ".png", 0)
    histogram = histogram_creating(image)
    cumsum = hist_cumsum(histogram)

    cdf = cumsum/max(cumsum)
    thresh_cyto = (np.where(cdf >= 0.5)[0][0])
    thresh_nucleus = (np.where(cdf >= 0.1)[0][0])

    print(f"For img {str(i)} the Thresh Cytoplasm: {thresh_cyto}")
    print(f"For img {str(i)} the Thresh Nucleus: {thresh_nucleus}")

    # # ----------------------------------------------------
    # # Plotting Histogram and CDF for visualization
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.bar(range(256), histogram, color='gray')
    # plt.axvline(x=thresh_cyto, color='blue', linestyle='--', label=f'Cyto Thresh = {thresh_cyto}')
    # plt.title(f"Histogram for Image {temp}")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(256), cdf, color='black')
    # plt.axhline(y=0.5, color='blue', linestyle='--', label='0.5 (Cyto Thresh)')
    # plt.title(f"CDF for Image {temp}")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("CDF")
    # plt.legend()
    #
    # plt.show()
    # # --------------------------------------------

    total_threshold_cyto = total_threshold_cyto + thresh_cyto
    total_threshold_nucleus = total_threshold_nucleus + thresh_nucleus

avg_threshold_cyto = total_threshold_cyto // 238
avg_threshold_nucleus = total_threshold_nucleus // 248

print(f"Avg Thresh Cyto: {avg_threshold_cyto}")
print(f"Avg Thresh Nucleus: {avg_threshold_nucleus}")

# #----------------------------------------------------------------
# #Code commented as it was just used for visualization
# #----------------------------------------------------------------
# image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/images/241.bmp",0)
# test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/masks/241.png",0)
#
# #Padding needs to be done as first row or first column may contain cytoplasm, which causes current CC to break
# cyto_img = padding(1,lower_by_x(image, avg_threshold_cyto))
# nucleus_img = padding(1, lower_by_x(image, avg_threshold_nucleus))
#
# cv.imshow('Image', image)
# cv.imshow('Mask', test_img)
# cv.imshow('Cytoplasm', cyto_img)
# cv.imshow('Nucleus', nucleus_img)
# cv.waitKey()
#
# #CCA
# cyto_img_cc, cyto_dic = cc(neg_img(cyto_img))
# nucleus_img_cc, nucleus_dict = cc(neg_img(nucleus_img))
#
# cyto_img_cc = remove_padding(cyto_img_cc, 1)
# nucleus_img_cc = remove_padding(nucleus_img_cc, 1)
#
# no_cyto = set(cyto_dic.values())
# no_nucleus = set(nucleus_dict.values())
#
# print(f"Cyto Dictionary Unique Values: {len(no_cyto)}")
# print(f"Nucleus Dictionary Unique Values: {len(no_nucleus)}")
# print(f"\nCyto Dict: {cyto_dic}")
# print(f"Nuclei Dict: {nucleus_dict}")
#
# print("\n\nNOW RUNNING GRAY AND WHITE SEPERATION FUNCTIONS")
#
# cyto_img_cc = cyto_to_gray(cyto_img_cc, cyto_dic)
# nucleus_img_cc = nuclei_to_white(nucleus_img_cc, nucleus_dict)
#
# own_mask = merge_for_mask(cyto_img_cc, nucleus_img_cc)
#
# cv.imshow('Image', image)
# cv.imshow('Mask', test_img)
# cv.imshow('Cytoplasm', cyto_img_cc)
# cv.imshow('Nucleus', nucleus_img_cc)
# cv.imshow('Own Mask', own_mask)
# cv.waitKey()
#
# print(np.unique(test_img))
# print(np.unique(own_mask))
# dc_black = calculate_dice_coefficient(test_img, own_mask, 0)
# dc_cyto = calculate_dice_coefficient(test_img, own_mask, 128)
# dc_nucleus = calculate_dice_coefficient(test_img, own_mask, 255)
#
# print(f"DC for Black: {dc_black}")
# print(f"DC for Cytoplasm: {dc_cyto}")
# print(f"DC for Nucleus: {dc_nucleus}")
# #------------------------------------------------------------------

print("\n\n--------------------------------NOW RUNNING ON TEST--------------------------------")
total_dc_black = 0
total_dc_cyto = 0
total_dc_nucleus = 0
for i in range (241, 301):
    if i < 10:
        temp = "00" + str(i)
    elif i < 100:
        temp = "0" + str(i)
    else:
        temp = str(i)

    image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/images/" + temp + ".bmp",0)  # Grayscale image
    test_img = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/masks/" + temp + ".png", 0)

    cyto_img = padding(1, lower_by_x(image, avg_threshold_cyto))
    nucleus_img = padding(1, lower_by_x(image, avg_threshold_nucleus))

    cyto_img_cc, cyto_dic = cc(neg_img(cyto_img))
    nucleus_img_cc, nucleus_dict = cc(neg_img(nucleus_img))

    cyto_img_cc = remove_padding(cyto_img_cc, 1)
    nucleus_img_cc = remove_padding(nucleus_img_cc, 1)

    cyto_img_cc = cyto_to_gray(cyto_img_cc, cyto_dic)
    nucleus_img_cc = nuclei_to_white(nucleus_img_cc, nucleus_dict)

    own_mask = merge_for_mask(cyto_img_cc, nucleus_img_cc)

    dc_black = calculate_dice_coefficient(test_img, own_mask, 0)
    dc_cyto = calculate_dice_coefficient(test_img, own_mask, 128)
    dc_nucleus = calculate_dice_coefficient(test_img, own_mask, 255)

    print(f"\nValues for img {str(i)}")
    print(f"DC for Black: {dc_black}")
    print(f"DC for Cytoplasm: {dc_cyto}")
    print(f"DC for Nucleus: {dc_nucleus}")

    total_dc_black += dc_black
    total_dc_cyto += dc_cyto
    total_dc_nucleus += dc_nucleus

avg_dc_black = total_dc_black / 60
avg_dc_cyto = total_dc_cyto / 60
avg_dc_nucleus = total_dc_nucleus / 60

print(f"\nAvg DC Black: {avg_dc_black}")
print(f"Avg DC Cyto: {avg_dc_cyto}")
print(f"Avg DC Nucleus: {avg_dc_nucleus}")