import numpy as np
import cv2 as cv

def padding(pad, orig):
    rows, cols = orig.shape
    padded_arr = np.ones((rows+ 2 * pad, cols+ 2 * pad), dtype = np.uint8)*0

    for i in range(rows):
        for j in range(cols):
            padded_arr[i+pad][j+pad] = orig[i][j]

    return padded_arr

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
                if (orig[i-1][j] == 0 and orig[i][j-1] == 0):
                    new_img[i][j] = count
                    my_dict[count] = count
                    count += 1
                elif (orig[i-1][j] == 255 and orig[i][j-1] == 0):
                    new_img[i][j] = find_root(my_dict, new_img[i-1][j])
                elif (orig[i-1][j] == 0 and orig[i][j-1] == 255):
                    new_img[i][j] = find_root(my_dict, new_img[i][j-1])
                elif (orig[i-1][j] == 255 and orig[i][j-1] == 255):
                    #This is to determine the smallest of the two values (left and up)
                    root1 = find_root(my_dict, new_img[i-1][j])
                    root2 = find_root(my_dict, new_img[i][j-1])
                    if root1 != root2:
                        if root1 < root2:
                            my_dict[root2] = root1
                        else:
                            my_dict[root1] = root2
                        new_img[i][j] = min(root1, root2)
                    else:
                        new_img[i][j] = root1

    for i in range(1, rows):
        for j in range(1, cols):
            if new_img[i][j] != 0:
                new_img[i][j] = find_root(my_dict, new_img[i][j])

    return new_img, my_dict

#Back tracks until it finds the root of that pixel i.e if we have a label of 1, 2 and 3 all in the same object, they all need to be corrected to point to 1
def find_root(my_dict, x):
    if x not in my_dict:
        my_dict[x] = x
        return x
    while my_dict[x] != x:
        my_dict[x] = my_dict[my_dict[x]]  # Path compression
        x = my_dict[x]
    return x

image = lower_by_2(padding(1, (cv.imread("D:/Uni/Semester 6/DIP/Self/Lab/Lab 2/cc.png", 0))))
cv.imshow('Window1',image)
cv.waitKey()

print(np.unique(image))
new_img, my_dictionary = cc(image)
print(my_dictionary)
new_img=new_img*40
no_obj = set(my_dictionary.values())
print(no_obj)
print(f"\nNo of objects: {len(no_obj)}")


cv.imshow('Window1', new_img)
cv.waitKey()