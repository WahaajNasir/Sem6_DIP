import cv2 as cv
import numpy as np
import math


def initial_centroids(image, k):
    unique_values = np.unique(image)

    init_centroids = np.random.choice(unique_values, k, replace=False)
    return np.array(init_centroids, dtype=np.float64)


def assign_cluster(image, centroids):
    rows, cols = image.shape
    labels = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            min_dist = float('inf')
            close_cluster = -1

            for index, centroid_intensity in enumerate(centroids):
                distance = abs(int(centroid_intensity) - int(image[i, j]))

                if distance < min_dist:
                    min_dist = distance
                    close_cluster = index

            labels[i, j] = close_cluster

    return labels


def update_centroids(image, labels, k):
    new_centroids = []

    for cluster in range(k):
        cluster_points = image[labels == cluster]

        if len(cluster_points) == 0:
            new_centroids.append(new_centroids[-1])
        else:
            new_centroid = np.mean(cluster_points)
            new_centroids.append(new_centroid)

    return np.array(new_centroids, dtype=np.float64)


def k_clustering(image, k, tolerance):
    centroids = initial_centroids(image, k)

    while True:
        labels = assign_cluster(image, centroids)
        new_centroids = update_centroids(image, labels, k)
        print("In this loop")

        converge = True
        for i in range(len(centroids)):
            distance = abs(centroids[i] - new_centroids[i])
            if distance > tolerance:
                converge = False
                break

        if converge:
            break

        centroids = new_centroids

    segmented_image = np.zeros_like(image)
    for cluster in range(k):
        segmented_image[labels == cluster] = int(255 / (k - 1)) * cluster

    print(np.unique(segmented_image))

    return segmented_image


# Main
image = cv.imread("D:/Uni/Semester 6/DIP/Self/Lec/Assignment 1/dataset_DIP_assignment/train/images/003.bmp", 0)
segmented_image = k_clustering(image, 3, 0)

cv.imshow("Original Image", image)
cv.imshow("K Means Segmentation", segmented_image)
cv.waitKey()
