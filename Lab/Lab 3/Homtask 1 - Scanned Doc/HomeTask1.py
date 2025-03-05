import cv2 as cv
import numpy as np
import fitz
from PIL import Image

def cc(orig, lower_bound, upper_bound):
    rows, cols = orig.shape
    new_img = np.zeros((rows, cols), dtype=np.uint16)
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

def mean_transform(image, mean):
    rows, cols = image.shape

    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if((image[i][j] >= 0) & (image [i][j] <= mean)):
                new_img[i][j] = 0
            elif((image[i][j] > mean) & (image [i][j] <= 255)):
                new_img[i][j] = 255

    return new_img

def power_transform(image):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            r = image[i][j]
            s = int(255 * ((r/255)**0.4))
            new_img[i][j] = s

    return new_img

def crop_img(image, x, y, h, w):
    return image[x:x + h, y:y + w]


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

def normalize_labels(image):
    max_label = np.max(image)
    if max_label > 255:
        scale_factor = 255.0 / max_label
        return (image * scale_factor).astype(np.uint8)
    return image.astype(np.uint8)

def remove_small_components(labeled_img, min_size):
    rows, cols = labeled_img.shape
    label_counts = {}

    # Count occurrences of each label
    for i in range(rows):
        for j in range(cols):
            label = labeled_img[i][j]
            if label > 0:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

    for i in range(rows):
        for j in range(cols):
            label = labeled_img[i][j]
            if label > 0 and label_counts[label] < min_size:
                labeled_img[i][j] = 0

    return labeled_img

def sep_alphabet(image):
    labels = np.unique(image)
    print(labels)

    for label in labels:
        if label == 0:
            continue

        x, y = np.where(image == label)
        x_min, x_max = int(min(x))-2, int(max(x))+2
        y_min, y_max = int(min(y))-2, int(max(y))+2

        temp = (crop_img(image, x_min, y_min, x_max - x_min, y_max - y_min))
        temp = (temp > 0).astype(np.uint8) * 255

        cv.imshow('Temp Window', temp)
        cv.waitKey()
        cv.destroyWindow('Temp Window')

def make_white(image):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype = np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i][j] > 0):
                new_img[i][j] = 255

    return new_img
#-------------------------------------------------------------
# Main code

doc = fitz.open("D:/Uni/Semester 6/DIP/Self/Lab/Lab 3/Lab 3/TestDoc.pdf")

# Convert the first page to an image
page = doc.load_page(0)  # First page (index 0)
pix = page.get_pixmap(colorspace=fitz.csGRAY)  # Convert to grayscale

# Convert to a NumPy array
image = np.array(Image.frombytes("L", [pix.width, pix.height], pix.samples))
mean = np.mean(image)
cv.imshow('Image', image)
cv.waitKey()

image = mean_transform(image, mean-150)
cv.imshow('Image', image)
cv.waitKey()

enhanced_img = power_transform(image)
cv.imshow('Enhanced', enhanced_img)
cv.waitKey()

cropped_img = crop_img(enhanced_img, 0, 0, 400, 800)
cv.imshow('Cropped Image', cropped_img)
cv.waitKey()

cropped_img = padding(1, cropped_img)
cc_cropped_img, cc_cropped_img_dict = cc(neg_img(cropped_img), 255, 255)
cc_cropped_img = remove_padding(cc_cropped_img, 1)

cc_cropped_img = normalize_labels(cc_cropped_img)
cv.imshow('Labeled', cc_cropped_img)
cv.waitKey()
print(set(cc_cropped_img_dict))

cc_cropped_img_rem = remove_small_components(cc_cropped_img, 20)
cv.imshow('Removed Small Components', cc_cropped_img_rem)
cv.waitKey()



image_again = make_white(cc_cropped_img_rem)
image_again = padding(1, image_again)
image_again_cc, _ = cc(image_again, 255, 255)
image_again_cc = remove_padding(image_again_cc, 1)

cv.imshow('New Image', image_again)

sep_alphabet(image_again_cc)
