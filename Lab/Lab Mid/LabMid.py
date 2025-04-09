import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pandas.compat.numpy.function import validate_argsort_kind


def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*0

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

def cc(orig, lower_bound, upper_bound):
    rows, cols = orig.shape
    new_img = np.zeros((rows, cols), dtype=np.int32)
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

def transformation_fun(cum_pdf):
    return np.uint8((cum_pdf*255))

def apply_trans(image, trans_fun):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            new_img[i][j] = trans_fun[image[i][j]]

    return new_img

def count_num(image, dict):
    rows, cols = image.shape
    count = 1
    max_val = 0

    new_img = np.zeros((rows, cols), dtype = np.uint8)

    value_arr = np.zeros(max_val, dtype= np.int32)
    for i in range(rows):
        for j in range(cols):
            val = image[i][j]
            value_arr[val] = value_arr[val]+1

    for i in range(len(value_arr)):
        val = value_arr[i]
        if val > 65:
            if val >= 200:
                for i in range(len(value_arr)):
                    if(i == val):
                        temp = i
                        break
                dict[temp] = 255
            else:
                for i in range(len(value_arr)):
                    if(i == val):
                        temp = i
                        break
                dict[temp] = 127
        else:
            dict[i] = 0

    for i in range(rows):
        for j in range(cols):
            val_img = image[i][j]
            for z in range(len(dict)):
                if z == val_img:
                    image[i][j] = dict(z)



# Main
image = cv.imread(r"D:\Uni\Semester 6\DIP\Self\Lab\Lab Mid\Open Lab with Instructions\dataset\images\render0358.png", 0)

image = contrast_stretch(image)
histogram = histogram_creating(image)
cumsum = hist_cumsum(histogram)
cdf = cumsum/max(cumsum)
thresh_rocks = (np.where(cdf >= 0.9)[0][0])
lowered_x = lower_by_x(image, thresh_rocks)

image_padded = padding(1, lowered_x)
img_cc_rocks, img_cc_rocks_dict = cc(image_padded, 255, 255)
count_num(img_cc_rocks, img_cc_rocks_dict)

cv.imshow("Original Image", image)
cv.imshow("Lowered", lowered_x)
cv.waitKey()

