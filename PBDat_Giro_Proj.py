import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import math
import os


# Description:
#   This function performs a Singular Value Decomposition (SVD) on the data matrix and reduces its dimensionality
# Inputs:
#   data_matrix -> original data matrix
#   rank -> rank of the reduced matrix
#   print_plots -> show plot of singular values
# Outputs:
#   data_redu -> reduced data matrix
def SVD_reduction(data_matrix, rank, print_plots):

    # SVD
    U, Sigma, Vh = np.linalg.svd(data_matrix, full_matrices=False)
    
    # Dimensionality reduction
    data_redu = (U[:, :rank] @ np.diag(Sigma)[:rank, :rank])
    print('SVD1: ', data_redu.shape)

    # Plot the singular values
    plt.figure(2)
    plt.plot(Sigma)
    plt.title('Singular values' + print_plots)
    plt.xlabel('Rank')
    
    return data_redu


# Description:
#   This function performs a Principal Component Analysis (PCA) on the data matrix and reduces its dimensionality
# Inputs:
#   data_matrix -> original data matrix
#   rank -> rank of the reduced matrix
#   print_plots -> show plot of explained variance
# Outputs:
#   data_redu -> reduced data matrix
def PCA_reduction(data_matrix, rank, print_plots):
    # Center data
    scaler = StandardScaler()
    data_scaled=scaler.fit_transform(data_matrix)

    # PCA
    pca_rank = PCA(n_components = rank)
    data_redu = pca_rank.fit_transform(data_scaled)
    print('PCA: ', data_redu.shape)

    # Plot the explained variance
    plt.figure()
    plt.plot(np.cumsum(pca_rank.explained_variance_ratio_))
    plt.title('Explained variance - ' + print_plots)
    plt.xlabel('Rank')

    return data_redu


# Description:
#   This function performs a Kmeans clustering on the data matrix
# Inputs:
#   data_matrix -> data matrix to cluster
#   n_cluster -> number of clusters to find
#   print_plots -> show plot of the clusters
# Outputs:
#   labels -> labels of the clusters
#   centroids -> centroids of the clusters
def kmeans_clustering(data_matrix, n_cluster, print_plots):
    
    #Kmeans
    kmeans = KMeans(n_clusters=n_cluster, n_init=10, random_state=10)
    kmeans.fit(data_matrix)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot the clusters 2D
    plt.figure()
    for i in range(n_cluster):
        plt.scatter(data_matrix[labels == i][:,0], data_matrix[labels == i][:,1], s=25, label='Label ' + str(i))
    plt.scatter(centroids[:,0], centroids[:,1], s=100, c='black', marker='x', label='Centroids')
    plt.legend()
    plt.title('Kmeans Clusters 2D - ' + print_plots)

    # Plot the clusters 3D
    ax = plt.figure().add_subplot(projection='3d')
    for i in range(n_cluster):
        ax.scatter(data_matrix[labels == i][:,0], data_matrix[labels == i][:,1], data_matrix[labels == i][:,2], s=25, label='Label ' + str(i))
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s=100, c='black', marker='x', label='Centroids')
    plt.legend()
    plt.title('Kmeans Clusters 3D - ' + print_plots)

    return labels, centroids


# Description:
#   This function performs an outlier detection on the data matrix
# Inputs:
#   data_matrix -> data matrix to find outliers
#   print_plots -> show plot of the outliers
# Outputs:
#   inliers -> inliers of the data matrix
#   outliers -> outliers of the data matrix
def outlier_detection(data_matrix, print_plots):
    
    # Z-score method
    z_score = stats.zscore(data_matrix, axis=0)
    dist = np.linalg.norm(z_score-data_matrix.mean(axis=0), axis=1)

    # Quantile of the distances
    q1 = np.quantile(dist, 0.25)
    q3 = np.quantile(dist, 0.75)
    iqr = q3 - q1

    # Detect outliers
    outlier_idx = np.where(dist > q3 + 1.5*iqr)[0]
    inlier_idx = np.where(dist <= q3 + 1.5*iqr)[0]

    # Remove outliers from data matrix
    inliers = np.delete(data_matrix, outlier_idx, axis=0)
    outliers = np.delete(data_matrix, inlier_idx, axis=0)

    plt.figure()
    plt.scatter(inliers[:,0], inliers[:,1], s=25, c='green', marker='o', label='Inliers')
    plt.scatter(outliers[:,0], outliers[:,1], s=40, c='red', marker='o', label='Outliers')
    plt.legend()
    plt.title('Outlier Detection - ' + print_plots)

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(inliers[:,0], inliers[:,1], inliers[:,2], s=25, c='green', marker='o', label='Inliers')
    ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], s=40, c='red', marker='o', label='Outliers')
    plt.legend()
    plt.title('Outlier Detection 3D - ' + print_plots)

    return inliers, outliers


# Description:
#   This function draws the skeleton on the frame image, linking their joints in case of high probability
# Inputs:
#   i -> frame number
#   skel_data -> skeleton coordinates and probabilities
#   frame -> frame image
#   width -> width of the frame
#   height -> height of the frame
def print_skel(i,skel_data,frame,width,height): #probably not done in the best way, i choose 0.35 experementaly
    val=4
    prob_val=0.20
    '''joint_edges = np.array([[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [0,14], [14,16], [0,15], [15,17]])
    for j in range(joint_edges.shape[0]):
        if skel_data[joint_edges[j, 0]*3, i] > prob_val and skel_data[joint_edges[j, 1]*3, i] > prob_val:
            cv2.line(img=frame, pt1=(int(skel_data[joint_edges[j, 0]*3+1,i]*width),int(skel_data[joint_edges[j, 0]*3+2,i]*height)), pt2=(int(skel_data[joint_edges[j, 1]*3+1,i]*width),int(skel_data[joint_edges[j, 1]*3+2,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    '''#0-1
    if skel_data[6,i]>prob_val and skel_data[3,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-2
    if skel_data[9,i]>prob_val and skel_data[6,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[7,i]*width),int(skel_data[8,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #2-3
    if skel_data[12,i]>prob_val and skel_data[9,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[7,i]*width),int(skel_data[8,i]*height)), pt2=(int(skel_data[10,i]*width),int(skel_data[11,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #3-4
    if skel_data[15,i]>prob_val and skel_data[12,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[10,i]*width),int(skel_data[11,i]*height)), pt2=(int(skel_data[13,i]*width),int(skel_data[14,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-5
    if skel_data[18,i]>prob_val and skel_data[6,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[16,i]*width),int(skel_data[17,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #5-6
    if skel_data[21,i]>prob_val and skel_data[18,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[16,i]*width),int(skel_data[17,i]*height)), pt2=(int(skel_data[19,i]*width),int(skel_data[20,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #6-7
    if skel_data[24,i]>prob_val and skel_data[21,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[19,i]*width),int(skel_data[20,i]*height)), pt2=(int(skel_data[22,i]*width),int(skel_data[23,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-8
    if skel_data[27,i]>prob_val and skel_data[6,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[25,i]*width),int(skel_data[26,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #8-9
    if skel_data[30,i]>prob_val and skel_data[27,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[25,i]*width),int(skel_data[26,i]*height)), pt2=(int(skel_data[28,i]*width),int(skel_data[29,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #9-10
    if skel_data[33,i]>prob_val and skel_data[30,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[28,i]*width),int(skel_data[29,i]*height)), pt2=(int(skel_data[31,i]*width),int(skel_data[32,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-11
    if skel_data[36,i]>prob_val and skel_data[6,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[34,i]*width),int(skel_data[35,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #11-12
    if skel_data[39,i]>prob_val and skel_data[36,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[34,i]*width),int(skel_data[35,i]*height)), pt2=(int(skel_data[37,i]*width),int(skel_data[38,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #12-13
    if skel_data[42,i]>prob_val and skel_data[39,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[37,i]*width),int(skel_data[38,i]*height)), pt2=(int(skel_data[40,i]*width),int(skel_data[41,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #0-14
    if skel_data[45,i]>prob_val and skel_data[3,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[43,i]*width),int(skel_data[44,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #14-16
    if skel_data[51,i]>prob_val and skel_data[45,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[43,i]*width),int(skel_data[44,i]*height)), pt2=(int(skel_data[49,i]*width),int(skel_data[50,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #0-15
    if skel_data[48,i]>prob_val and skel_data[3,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[46,i]*width),int(skel_data[47,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #15-17
    if skel_data[54,i]>prob_val and skel_data[48,i]>prob_val:
        cv2.line(img=frame, pt1=(int(skel_data[46,i]*width),int(skel_data[47,i]*height)), pt2=(int(skel_data[52,i]*width),int(skel_data[53,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #'''


# Description:
#   This function fills the missing values of the data matrix with the mean
# Inputs:
#   skel_data -> incomplete skeleton matrix
# Outputs:
#   mask_miss -> mask of missing values
#   n_missing -> number of missing values

def fill_missing(skel_data):
    mask_miss = np.where(skel_data == 0, 1, 0)
    n_missing = np.count_nonzero(mask_miss)
    return mask_miss, n_missing


# Description:
#   This function performs a completion of the skeletons matrix using SVD
#   It also centers and normalizes the matrix
# Inputs:
#   skel_incomp -> incomplete skeletons matrix
#   skel_comp -> complete skeletons matrix
#   rank -> rank of matrix, can be higher or equal, the closest the better
#   alfa -> error for which we consider that it has already converged to a satisfactory solution
# Outputs:
#   skel -> completed matrix
#   missing_index -> index of missing values from original incomplete matrix
#   it -> number of iterations done to acomplish result
def fill_missing_alg(skel,OG_skel,r,alfa):
    original_data=np.copy(skel)
    skel=np.delete(skel,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54],0) #we take off index and probabilities
    OG_skel=np.delete(OG_skel,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54],0) #we take off index and probabilities
    missing_index,n=fill_missing(skel) #get missing indexes and fill in with something (can be mean, or just some random number)
    alfa=alfa/n
    err_dif=math.inf
    last_error=1000
    it=0
    old_skel=np.copy(skel)
    while err_dif>alfa and it<200 : # this will stop at either some iteration or when the SVD stabalizes
        norm_val_x=np.zeros(skel[0,:].shape[0])
        norm_val_y=np.zeros(skel[0,:].shape[0])
        sub_val_x=np.zeros(skel[0,:].shape[0])
        sub_val_y=np.zeros(skel[0,:].shape[0])
        for i in range(skel[0,:].shape[0]): #get maximum value
            for j in range(skel[:,0].shape[0]):
                if j%2==0:
                    if skel[j,i]>norm_val_x[i]:
                        norm_val_x[i]=np.copy(skel[j,i])
                else:
                    if skel[j,i]>norm_val_y[i]:
                        norm_val_y[i]=np.copy(skel[j,i])
        #normalize and centralize data

        for i in range(skel[0,:].shape[0]):
            for j in range(skel[:,0].shape[0]):
                if j%2==0:
                    skel[j,i]=skel[j,i]/norm_val_x[i]
                    if j==0:
                        sub_val_x[i]=skel[j,i]
                    skel[j,i]=skel[j,i]-sub_val_x[i]
                else:
                    skel[j,i]=skel[j,i]/norm_val_y[i]
                    if j==1:
                        sub_val_y[i]=skel[j,i]
                    skel[j,i]=skel[j,i]-sub_val_y[i]
        #get prediction

        u, s, v = np.linalg.svd(skel, full_matrices=False) #SVD
        s=np.diag(s)
        skel_pred =  u[:, :r] @ s[0:r, 0:r] @ v[0:r, :] #reconstruct
        for i in range(skel[0,:].shape[0]):
            for j in range(skel[:,0].shape[0]):
                if missing_index[j,i]==1:
                    skel[j,i]=np.copy(skel_pred[j,i]) 
        #return to its place in the video

        for i in range(skel[0,:].shape[0]):
            for j in range(skel[:,0].shape[0]):
                if j%2==0:
                    skel[j,i]=skel[j,i]+sub_val_x[i]
                    skel[j,i]=skel[j,i]*norm_val_x[i]
                else:
                    skel[j,i]=skel[j,i]+sub_val_y[i]
                    skel[j,i]=skel[j,i]*norm_val_y[i] 
        #calculate error between iterations, this isint to show that its getting "better", but to show that its converging

        err=0
        n_err=0
        for i in range(skel[0,:].shape[0]): #get error, which is the distance of each predicted point to its previous predictions
            for j in range(skel[:,0].shape[0]):
                if missing_index[j,i]==1 and j%2==0:
                    aux=float((old_skel[j,i]-skel[j,i])*(old_skel[j+1,i]-skel[j+1,i]))
                    err+=math.sqrt(aux*aux)
                    n_err+=1
        final_error=err/n_err #we will use the normalized error because its easier to analize
        err_dif=last_error-final_error
        if err_dif<0: #neglect negative errors
            err_dif=100000000000000000
        last_error=final_error
        old_skel=np.copy(skel)
        it+=1
        print("Did iteration: ",it)
        print("error: ",final_error)   
    #since we have the complete skelectons, we can also have an "absolute" error

    err=0
    n_err=0
    for i in range(skel[0,:].shape[0]): #get error, no real meaning, only for comparations sake
        for j in range(skel[:,0].shape[0]):
            if missing_index[j,i]==1 and j%2==0:
                aux=float((OG_skel[j,i]-skel[j,i])*(OG_skel[j+1,i]-skel[j+1,i]))
                err+=math.sqrt(aux*aux)
                n_err+=1
    final_error=err/n_err
    print("Error relative to the complete skelectons: ",final_error)
    for i in range(18): #return to its normal shape
        original_data[(i*3)+1,:]=np.copy(skel[(i*2),:])
        original_data[(i*3)+2,:]=np.copy(skel[(i*2)+1,:])
    skel=np.copy(original_data)
    return skel,missing_index,it

# Print flags
print_plots = True
print_frames = True
print_data = False
manual_kmeans = True
subset=True

# Frames to select
frame_start = 5000
frame_count = 1000
var = 0.85

# Path to save frame images
path = 'C:/Users/Tiago/Desktop/Project_PBDat/frames'

# Loading video
vid = cv2.VideoCapture('data/girosmallveryslow2.mp4')
n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Loading skeletons incomplete
skel=scipy.io.loadmat('data/girosmallveryslow2_openpose.mat')
skel=skel['skeldata']

# Loading skeletons completed
skel_comp=scipy.io.loadmat('data/girosmallveryslow2_openpose_complete.mat')
skel_comp=skel_comp['skeldata']

# Loading embeddings
features = scipy.io.loadmat('data/girosmallveryslow2.mp4_features.mat')
features = features['features'] 

# Select a subset of frames
if subset:
    features = features[:, frame_start:frame_start + frame_count] # select a subset of 100 frames
    features = features.T

    skel[:, (skel[0, :] >= frame_start) & (skel[0, :] < frame_start + frame_count)]
    skel_comp[:, (skel_comp[0, :] >= frame_start) & (skel_comp[0, :] < frame_start + frame_count)]

#######################
#  SKELETON ANALYSIS  #
#######################

#Fill missing data
#skel_new,missing_index,it=fill_missing_alg(skel,skel_comp,4,0.001) 
#print("This took ",it," iterations")

skel_redu = PCA_reduction(skel_comp, 15, 'Skeletons')

# Clustering of the features
labels, centroid = kmeans_clustering(skel_redu, 4, 'Skeletons')
inliers, outliers = outlier_detection(skel_redu, 'Skeletons')


#######################
#  FEATURES ANALYSIS  #
#######################

# SVD
# TODO: Find the rank in a better way
U, S, Vt = np.linalg.svd(features, full_matrices=False)
Variance = S**2 / np.sum(S**2)
Variance_sum = np.cumsum(Variance) / np.sum(Variance)

# Find rank of total variance > N%
for i in range(len(Variance_sum)):
    if Variance_sum[i] > var:
        rank = i
        print('rank ( >', var, '):', rank)
        break

# Plot the sum of the variance
plt.figure(1)
plt.plot(Variance_sum)
plt.title('Sum of the variance')
plt.xlabel('Rank')

# Dimensionality reduction of the features
#features_redu = SVD_reduction(features, rank, 'Features')
features_redu = PCA_reduction(features, rank, 'Features')

# Clustering of the features
labels, centroid = kmeans_clustering(features_redu, 4, 'Features')
inliers, outliers = outlier_detection(features_redu, 'Features')
vid_skel_ind=skel[0,:]
#vid_skel_new=skel_new[0,:]

#######################
#  ALL DATA ANALYSIS  #
#######################

# TODO: Combine both matrices and cluster them

# Showing video
if print_frames:
    for i in range(frame_count):

        # Read the frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_start + i)
        res, frame = vid.read()

        # Print the skeleton on the frame
        skel_frame=np.where(vid_skel_ind==i+frame_start-1)
        #skel_new_frame=np.where(vid_skel_new==i+frame_start-1)
        for j in skel_frame[0]:
           print_skel(j ,skel,frame,frame.shape[1],frame.shape[0])
        #for j in skel_new_frame[0]:
        #    print_skel(j ,skel_new,frame,frame.shape[1],frame.shape[0])
        
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

        # Save the frame image in the path directory
        # The name of the image is the label of its cluster and its frame number
        cv2.imwrite(os.path.join(path, str(labels[i]) + '_frame' + str(frame_start + i) + '.png'), frame)
        
        if cv2.waitKey(0) == ord('q'):
            break

    #stop functions
    vid.release()
    cv2.destroyAllWindows()

if print_plots:
    plt.show()