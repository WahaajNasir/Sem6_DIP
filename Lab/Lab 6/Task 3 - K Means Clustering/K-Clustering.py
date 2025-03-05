import cv2 as cv
import numpy as np
import math

def initial_centroids(image, k):
    rows, cols = image.shape
    all_indices = []

    for i in range(rows):
        for j in range(cols):
            all_indices.append((i, j, image[i, j]))  # Include intensity

    all_indices = np.array(all_indices, dtype=np.float64)
    init_centroids = all_indices[np.random.choice(len(all_indices), k, replace=False)]

    return init_centroids

def assign_cluster(image, centroids):
    rows, cols = image.shape
    labels = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            min_dist = float('inf')
            close_cluster = -1

            for index, (cx, cy, cint) in enumerate(centroids):  # Include intensity
                distance = math.sqrt((cx - i) ** 2 + (cy - j) ** 2 + (cint - image[i, j]) ** 2)

                if distance < min_dist:
                    min_dist = distance
                    close_cluster = index

            labels[i, j] = close_cluster

    return labels

def update_centroids(image, labels, k):
    new_centroids = []
    rows, cols = image.shape

    for cluster in range(k):
        cluster_points = []

        for i in range(rows):
            for j in range(cols):
                if labels[i, j] == cluster:
                    cluster_points.append((i, j, image[i, j]))  # Include intensity

        if len(cluster_points) == 0:
            new_centroids.append(new_centroids[-1])  # Keep previous centroid
        else:
            cluster_points = np.array(cluster_points, dtype=np.float64)
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(tuple(new_centroid))

    return np.array(new_centroids, dtype=np.float64)

def k_clustering(image, k, tolerance):
    rows, cols = image.shape
    centroids = initial_centroids(image, k)

    while True:
        labels = assign_cluster(image, centroids)
        new_centroids = update_centroids(image, labels, k)
        print("In this loop")

        converge = True
        for i in range(len(centroids)):
            distance = math.sqrt((centroids[i][0] - new_centroids[i][0]) ** 2 +
                                 (centroids[i][1] - new_centroids[i][1]) ** 2 +
                                 (centroids[i][2] - new_centroids[i][2]) ** 2)  # Include intensity

            if distance > tolerance:
                converge = False
                break

        if converge:
            break

        centroids = new_centroids

    segmented_image = np.zeros((rows, cols), dtype=np.uint8)
    for cluster in range(k):
        segmented_image[labels == cluster] = int(255 / (k - 1)) * cluster

    print(np.unique(segmented_image))

    return segmented_image


# Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/003.bmp", 0)
segmented_image = k_clustering(image, 3, 5)

cv.imshow("Original Image", image)
cv.imshow("K Means Segmentation", segmented_image)
cv.waitKey()
