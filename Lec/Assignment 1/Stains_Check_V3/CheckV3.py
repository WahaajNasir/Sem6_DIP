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

# def lower_by_x(image, thresh):
#     rows, cols = image.shape
#     new_image = np.ones((rows, cols), dtype=np.uint8)
#
#     for i in range(rows):
#         for j in range(cols):
#             if (image[i, j] >= 0 and image[i, j] <= thresh):
#                 new_image[i, j] = 0
#             elif (image[i, j] >= thresh+1 and image[i, j] <= 255):
#                 new_image[i, j] = 255
#
#     return new_image
#
# def lower_by_2(image):
#     rows, cols = image.shape
#     new_image = np.ones((rows, cols), dtype=np.uint8)
#
#     for i in range(rows):
#         for j in range(cols):
#             if (image[i, j] >= 0 and image[i, j] <= 127):
#                 new_image[i, j] = 0
#             elif (image[i, j] >= 128 and image[i, j] <= 255):
#                 new_image[i, j] = 255
#
#     return new_image

def cc(orig, lower_bound, upper_bound):
    rows, cols = orig.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8)
    my_dict = {}
    count = 1

    for i in range(1, rows):
        for j in range(1, cols):
            if ((orig[i][j] >= lower_bound) & (orig[i][j] <= upper_bound)) :
                neighbors = []  # Store nonzero neighboring labels

                # Check all 8-connected neighbors
                if ((orig[i - 1][j] >= lower_bound) & (orig[i - 1][j] <= upper_bound)):
                    neighbors.append(new_img[i - 1][j])
                if ((orig[i][j-1] >= lower_bound) & (orig[i][j-1] <= upper_bound)):
                    neighbors.append(new_img[i][j - 1])
                if ((orig[i-1][j-1] >= lower_bound) & (orig[i-1][j-1] <= upper_bound)):
                    neighbors.append(new_img[i - 1][j - 1])
                if ((j + 1 < cols) and (lower_bound <= orig[i - 1][j + 1] <= upper_bound)):
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

def overlapping_labels(cyto_img, nucleus_img):
    rows, cols = cyto_img.shape
    new_cyto = np.zeros((rows, cols), dtype = np.uint8)
    new_nucleus = np.zeros((rows, cols), dtype = np.uint8)

    cyto_labels_keep = set()
    nucleus_labels_keep = set()

    for i in range(rows):
        for j in range(cols):

            #Check for overlapping areas
            if((cyto_img[i][j] > 0) & (nucleus_img[i][j] > 0)):
                cyto_labels_keep.add(cyto_img[i][j])
                nucleus_labels_keep.add(nucleus_img[i][j])

    for i in range(rows):
        for j in range(cols):

            if (cyto_img[i][j] in cyto_labels_keep):
                new_cyto[i][j] = cyto_img[i][j]

            if(nucleus_img[i][j] in nucleus_labels_keep):
                new_nucleus[i][j] = nucleus_img[i][j]

    return new_cyto, new_nucleus, cyto_labels_keep, nucleus_labels_keep

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

def contrast_stretch(image):
    im_min_5 = np.percentile(image, 5)
    im_max_95 = np.percentile(image, 95)
    rows,cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if(image[i][j] < im_min_5):
                new_img[i][j] = 0
            elif(image[i][j] > im_max_95):
                new_img[i][j] = 255
            else:
                new_img[i][j] = 255 * ((image[i][j] - im_min_5) / (im_max_95 - im_min_5))

    return new_img


#Main


best_dc_black = 0
best_dc_cyto = 0
best_dc_nucleus = 0
best_thresh_cyto = 0
best_thresh_nucleus = 0

for thresh_cyto in np.arange(0.4, 0.8, 0.05):  # Iterate from 0.4 to 0.8 with step 0.005
    for thresh_nucleus in np.arange(0.05, 0.25, 0.05):  # Iterate from 0.05 to 0.25 with step 0.005
        total_threshold_cyto = 0
        total_threshold_nucleus = 0

        # ---------------- TRAINING PHASE ----------------
        for img_id in range(3, 201):  # Only train on images 3 to 200
            temp = str(img_id).zfill(3)  # Ensures correct file naming format (e.g., "003", "198")

            image = cv.imread(
                f"D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/{temp}.bmp", 0)
            test_img = cv.imread(
                f"D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/masks/{temp}.png", 0)

            image = contrast_stretch(image)
            histogram = histogram_creating(image)
            cumsum = hist_cumsum(histogram)

            cdf = cumsum / max(cumsum)
            computed_thresh_cyto = np.where(cdf >= thresh_cyto)[0][0]
            computed_thresh_nucleus = np.where(cdf >= thresh_nucleus)[0][0]

            total_threshold_cyto += computed_thresh_cyto
            total_threshold_nucleus += computed_thresh_nucleus

        avg_threshold_cyto = total_threshold_cyto // 198  # Total training images = 198
        avg_threshold_nucleus = total_threshold_nucleus // 198

        # ---------------- VALIDATION PHASE ----------------
        total_dc_black = 0
        total_dc_cyto = 0
        total_dc_nucleus = 0

        for img_id in range(201, 241):  # Only validate on images 201 to 240
            temp = str(img_id).zfill(3)

            image = cv.imread(
                f"D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/images/{temp}.bmp", 0)
            test_img = cv.imread(
                f"D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/test/masks/{temp}.png", 0)

            image = contrast_stretch(image)
            histogram = histogram_creating(image)
            cumsum = hist_cumsum(histogram)

            cdf = cumsum / max(cumsum)
            computed_thresh_cyto = np.where(cdf >= thresh_cyto)[0][0]
            computed_thresh_nucleus = np.where(cdf >= thresh_nucleus)[0][0]

            # Taking the average of computed and trained thresholds
            final_thresh_cyto = (avg_threshold_cyto + computed_thresh_cyto) // 2
            final_thresh_nucleus = (avg_threshold_nucleus + computed_thresh_nucleus) // 2

            image_padded = padding(1, image)
            img_cc_cyto, img_cc_cyto_dict = cc(image_padded, 0, final_thresh_cyto)
            img_cc_nucleus, img_cc_nucleus_dict = cc(image_padded, 0, final_thresh_nucleus)
            image_cc_cyto = remove_padding(img_cc_cyto, 1)
            image_cc_nucleus = remove_padding(img_cc_nucleus, 1)

            image_cc_cyto, image_cc_nucleus, img_cc_cyto_dict, img_cc_nucleus_dict = overlapping_labels(
                image_cc_cyto, image_cc_nucleus)
            image_cc_cyto = cyto_to_gray(img_cc_cyto, img_cc_cyto_dict)
            image_cc_nucleus = nuclei_to_white(img_cc_nucleus, img_cc_nucleus_dict)

            own_mask = merge_for_mask(image_cc_cyto, image_cc_nucleus)

            # Calculate Dice Coefficients for all three classes
            dc_black = calculate_dice_coefficient(test_img, own_mask, 0)
            dc_cyto = calculate_dice_coefficient(test_img, own_mask, 128)
            dc_nucleus = calculate_dice_coefficient(test_img, own_mask, 255)

            total_dc_black += dc_black
            total_dc_cyto += dc_cyto
            total_dc_nucleus += dc_nucleus

        avg_dc_black = total_dc_black / 40  # 40 validation images
        avg_dc_cyto = total_dc_cyto / 40
        avg_dc_nucleus = total_dc_nucleus / 40

        # ---------------- TRACKING BEST RESULTS ----------------
        if avg_dc_black > best_dc_black:
            best_dc_black = avg_dc_black

        if avg_dc_cyto > best_dc_cyto:
            best_dc_cyto = avg_dc_cyto
            best_thresh_cyto = thresh_cyto

        if avg_dc_nucleus > best_dc_nucleus:
            best_dc_nucleus = avg_dc_nucleus
            best_thresh_nucleus = thresh_nucleus

        print(f"Thresh Cyto: {thresh_cyto}, Thresh Nucleus: {thresh_nucleus}")
        print(f"Avg DC Black: {avg_dc_black}, Avg DC Cyto: {avg_dc_cyto}, Avg DC Nucleus: {avg_dc_nucleus}\n")

# ---------------- FINAL RESULTS ----------------
print("\nBest results found:")
print(f"Best Threshold Cyto: {best_thresh_cyto}, Best DC Cyto: {best_dc_cyto}")
print(f"Best Threshold Nucleus: {best_thresh_nucleus}, Best DC Nucleus: {best_dc_nucleus}")
print(f"Best DC for Background (Black): {best_dc_black}")