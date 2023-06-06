import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def print_skel(i,skel_data,frame,width,height): #prints skel onto video, prob_val is calculated experimentaly
    val=4
    prob_val=0.20
    #0-1
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

def fill_missing(skel_data):
    #Desc:
    #   fill missing values with the mean
    #Inputs:
    #   skel_data->data matrix
    #Outputs:
    #   missing_index->index of missing values from incomplete matrix 
    #   n->nº of missing values
    n_row=skel_data[:,0].shape
    n_col=skel_data[0,:].shape
    missing_index=np.zeros((n_row[0],n_col[0]))
    for i in range(n_row[0]):
        sum=0
        n=0
        for j in range(n_col[0]):
            if skel_data[i,j]==0:
                missing_index[i][j]=1
            else:
                sum+=skel_data[i,j]
                n+=1
        for j in range(n_col[0]):
            if missing_index[i][j]==1:

                if n!=0:
                    skel_data[i,j]=0
    return missing_index,n

def fill_missing_alg(skel,OG_skel,r,alfa): #matrix completion algoritm
    #Desc:
    #   This will use SVD to complete a matrix with missing values. To do this it will also center and normalize the datamatrix
    #Inputs:
    #   skel->matrix to be completed
    #   OG_skel->matrix thats already completed for comparation
    #   r-> rank of matrix, can be higher or equal, the closest the better
    #   alfa-> error for which we consider that it has already converged to a satisfactory solution
    #Outputs:
    #   skel->completed matrix
    #   missing_index->index of missing values from original incomplete matrix
    #   it-> nº of iterations done to acomplish result

    original_data=np.copy(skel)
    skel=np.delete(skel,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54],0) #we take off index and probabilities
    OG_skel=np.delete(OG_skel,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54],0) #we take off index and probabilities
    missing_index,n=fill_missing(skel) #get missing indexes and fill in with something (can be mean, or just some random number)
    alfa=alfa/n
    err_dif=10000000000000000000000000;
    last_error=1000;
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
manual_kmeans = False

# Frames to select
frame_start = 5000
frame_count = 1000
var = 0.85


# Loading video
vid = cv2.VideoCapture('data/girosmallveryslow2.mp4')
n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

#skelecton init
skel=scipy.io.loadmat('data/girosmallveryslow2_openpose.mat')
skel=skel['skeldata']
skel_comp=scipy.io.loadmat('data/girosmallveryslow2_openpose_complete.mat')
skel_comp=skel_comp['skeldata']

# Loading embeddings
data = scipy.io.loadmat('data/girosmallveryslow2.mp4_features.mat')
data = data['features'] 

# Select a subset of frames
subset = data[:, frame_start:frame_start + frame_count] # select a subset of 100 frames
subset = subset.T

#Fill missing data

skel,missing_index,it=fill_missing_alg(skel,skel_comp,4,0.001) 
print("This took ",it," iterations")

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