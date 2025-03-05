import cv2
import numpy as np

def image_read(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = lower_by_2(img)
    print(np.unique(img))

    if img is None:
        print("Error: Could not read the image.")
        return
    return  img


def show_image(matrix,x ,windowName):
    padded_matrix = np.array(matrix, dtype=np.uint8) *x
    print('padded matrix: ', padded_matrix.shape)
    cv2.imshow(windowName, padded_matrix)
    cv2.waitKey(0)
    cv2.destroyWindow(windowName)

def padded_image(img):
    height, width = img.shape[:2]
    padded_matrix = np.ones((height + 2, width + 2), dtype=np.uint8)*0
    padded_matrix[1:-1, 1:-1] = img
    return padded_matrix

def find_root(eq_list, key):
    if key not in eq_list:
        eq_list[key]=key
        return key
    while eq_list[key] != key:
        eq_list[key] = int(eq_list[eq_list[key]])  # Path compression
        key = eq_list[key]
    return key
def apply_path_compression(eq_list):
    for key in eq_list.keys():
        root = find_root(eq_list, key)
        eq_list[key] = root
def connectivit_4(image):
    height, width = image.shape
    connect_4 = np.zeros((height, width), dtype=np.uint8)
    eq_list = {}
    label = 1

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i, j] == 255:  # Foreground pixel
                left = connect_4[i, j - 1]
                top = connect_4[i - 1, j]

                if left == 0 and top == 0:
                    connect_4[i, j] = label
                    eq_list[label] = label
                    label += 1
                elif left != 0 and top!=0:
                    root_label = find_root(eq_list, left)
                    min_label = min(left, top)
                    max_label = max(left, top)
                    connect_4[i, j] = find_root(eq_list, min_label)
                    eq_list[find_root(eq_list, max_label)] = connect_4[i, j]
                else:
                    connect_4[i, j] = find_root(eq_list, max(left, top))

    for key in eq_list.keys():
        eq_list[key] = find_root(eq_list, key)

    for i in range(1,height-1):
        for j in range(1,width-1):
            if connect_4[i][j] != 0:
                connect_4[i, j] = find_root(eq_list,eq_list[connect_4[i, j]])
    print("Number of labels:", len(eq_list))
    print("Equivalence list:", eq_list)
    print(f"Unique Vals: {set(eq_list.values())}")
    return connect_4, eq_list

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

img=image_read("D:/Uni/Semester 6/DIP/Self/Lab/Lab 2/cc.png")
pad_img=padded_image(img)
show_image(pad_img,1,"Padded Image")
#print(pad_img)
img_4, eq_list=connectivit_4(pad_img)
print(f"No of Objects: {len(set(eq_list.values()))}")
show_image(img_4,255," connect_4")