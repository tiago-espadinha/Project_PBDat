import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Print flags
print_plots = True
print_frames = True
print_data = False
manual_kmeans = False

# Frames to select
frame_start = 5000
frame_count = 1000
var = 0.85


# Loading video
vid = cv2.VideoCapture('data/girosmallveryslow2.mp4')
n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Loading embeddings
data = scipy.io.loadmat('data/girosmallveryslow2.mp4_features.mat')
data = data['features'] 

# Select a subset of frames
subset = data[:, frame_start:frame_start + frame_count] # select a subset of 100 frames
subset = subset.T


# PCA Manual

# Center data
mu = subset.mean(axis=0)
subset_zc = subset - mu

# SVD
U, S, Vt = np.linalg.svd(subset_zc, full_matrices=False)
Variance = S**2 / np.sum(S**2)
Variance_sum = np.cumsum(Variance) / np.sum(Variance)
if print_data:
    print(Variance)
    print(U.shape, S.shape, Vt.shape)
    print('S:', S)
    print(np.linalg.norm(subset_zc - (U * S) @ Vt))
    print('Variance_sum:', Variance_sum)

# Find rank of total variance > N%
for i in range(len(Variance_sum)):
    if Variance_sum[i] > var:
        rank = i
        print('rank ( >', var, '):', rank)
        break

subset_zcr = (U[:,:rank] * S[:rank]) @ Vt[:rank,:]
print((np.linalg.norm(subset_zc - subset_zcr), np.linalg.norm(subset_zc)))
subset_r = subset_zcr + mu

# Plot the singular values
plt.figure(1)
plt.plot(S)
plt.title('Singular values')
plt.xlabel('Rank')

# Plot the sum of the variance
plt.figure(2)
plt.plot(Variance_sum)
plt.title('Sum of the variance')
plt.xlabel('Rank')


# PCA Sklearn

# Center data
scaler = StandardScaler()
scaler.fit(subset)
subset_scaled = scaler.transform(subset)

# PCA
pca_rank = PCA(n_components = rank)
pca_rank.fit(subset_scaled)
subset_pca = pca_rank.transform(subset_scaled)

# Plot the explained variance ratio
'''
plt.figure(3)
plt.plot(np.cumsum(pca_rank.explained_variance_ratio_))
plt.title('Explained variance ratio')
'''


#Kmeans

kmeans = KMeans(n_clusters=4, n_init=10)
if manual_kmeans:
    kmeans.fit(subset_r)
else:
    kmeans.fit(subset_pca)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

if manual_kmeans:
    cluster1 = subset_r[labels == 0]
    cluster2 = subset_r[labels == 1]
    cluster3 = subset_r[labels == 2]
    cluster4 = subset_r[labels == 3]
else:
    cluster1 = subset_pca[labels == 0]
    cluster2 = subset_pca[labels == 1]
    cluster3 = subset_pca[labels == 2]
    cluster4 = subset_pca[labels == 3]

# Plot the clusters
plt.figure(4)
plt.scatter(cluster1[:,0], cluster1[:,1], s=25, c='b')
plt.scatter(cluster2[:,0], cluster2[:,1], s=25, c='g')
plt.scatter(cluster3[:,0], cluster3[:,1], s=25, c='r')
plt.scatter(cluster4[:,0], cluster4[:,1], s=25, c='y')
plt.scatter(centroids[:,0], centroids[:,1], s=100, c='black', marker='x')
plt.legend(['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Centroids'])
plt.title('Kmeans Clusters')

ax = plt.figure(5).add_subplot(projection='3d')
ax.scatter(cluster1[:,0], cluster1[:,1], cluster1[:,2], s=25, c='b')
ax.scatter(cluster2[:,0], cluster2[:,1], cluster2[:,2], s=25, c='g')
ax.scatter(cluster3[:,0], cluster3[:,1], cluster3[:,2], s=25, c='r')
ax.scatter(cluster4[:,0], cluster4[:,1], cluster4[:,2], s=25, c='y')
ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s=100, c='black', marker='x')

plt.legend(['Label 0', 'Label 1', 'Label 2', 'Label 3', 'Centroids'])
plt.title('Kmeans Clusters 3d')

# Showing video
if print_frames:
    for i in range(frame_count):

        # Choose some frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_start + i)
        res, frame = vid.read()

        # Set the text properties
        label_text = str(labels[i])
        frame_text = str(frame_start + i)
        bottom_right_corner = (frame.shape[1] - 30, frame.shape[0] - 20)
        bottom_left_corner = (30, frame.shape[0] - 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        line_thickness = 2

        # Add the frame and cluster number to the image
        cv2.putText(frame, frame_text, bottom_left_corner, font, font_scale, font_color, line_thickness)
        cv2.putText(frame, label_text, bottom_right_corner, font, font_scale, font_color, line_thickness)
        cv2.imshow('Video Frames', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    #stop functions
    vid.release()
    cv2.destroyAllWindows()

if print_plots:
    plt.show()