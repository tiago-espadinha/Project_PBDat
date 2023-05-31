import numpy as np
import scipy
import scipy.io
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def print_skel(i,skel_data,frame,width,height): #probably not done in the best way, i choose 0.35 experementaly
    val=4
    #0-1
    if skel_data[6,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-2
    if skel_data[9,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[7,i]*width),int(skel_data[8,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #2-3
    if skel_data[12,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[7,i]*width),int(skel_data[8,i]*height)), pt2=(int(skel_data[10,i]*width),int(skel_data[11,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #3-4
    if skel_data[15,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[10,i]*width),int(skel_data[11,i]*height)), pt2=(int(skel_data[13,i]*width),int(skel_data[14,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-5
    if skel_data[18,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[16,i]*width),int(skel_data[17,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #5-6
    if skel_data[21,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[16,i]*width),int(skel_data[17,i]*height)), pt2=(int(skel_data[19,i]*width),int(skel_data[20,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #6-7
    if skel_data[24,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[19,i]*width),int(skel_data[20,i]*height)), pt2=(int(skel_data[22,i]*width),int(skel_data[23,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-8
    if skel_data[27,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[25,i]*width),int(skel_data[26,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #8-9
    if skel_data[30,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[25,i]*width),int(skel_data[26,i]*height)), pt2=(int(skel_data[28,i]*width),int(skel_data[29,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #9-10
    if skel_data[33,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[28,i]*width),int(skel_data[29,i]*height)), pt2=(int(skel_data[31,i]*width),int(skel_data[32,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #1-11
    if skel_data[36,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[4,i]*width),int(skel_data[5,i]*height)), pt2=(int(skel_data[34,i]*width),int(skel_data[35,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #11-12
    if skel_data[39,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[34,i]*width),int(skel_data[35,i]*height)), pt2=(int(skel_data[37,i]*width),int(skel_data[38,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #12-13
    if skel_data[42,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[37,i]*width),int(skel_data[38,i]*height)), pt2=(int(skel_data[40,i]*width),int(skel_data[41,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #0-14
    if skel_data[45,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[43,i]*width),int(skel_data[44,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #14-16
    if skel_data[51,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[43,i]*width),int(skel_data[44,i]*height)), pt2=(int(skel_data[49,i]*width),int(skel_data[50,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #0-15
    if skel_data[48,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[1,i]*width),int(skel_data[2,i]*height)), pt2=(int(skel_data[46,i]*width),int(skel_data[47,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)
    #15-17
    if skel_data[54,i]>0.35:
        cv2.line(img=frame, pt1=(int(skel_data[46,i]*width),int(skel_data[47,i]*height)), pt2=(int(skel_data[52,i]*width),int(skel_data[53,i]*height)), color=(255, 0, 0), thickness=val , lineType=8, shift=0)


#video init
vid=cv2.VideoCapture('../../../data/girosmallveryslow2.mp4')
n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number=5895

#matrix init
data=scipy.io.loadmat('../../../data/girosmallveryslow2.mp4_features.mat')
data=data['features'] 
base=data[:,frame_number:frame_number+11] #define base->linearly depedent

#skelecton init
skel=scipy.io.loadmat('../../../data/girosmallveryslow2_openpose_complete.mat')
skel=skel['skeldata']


#EDA stuff
#get span
span, s, vh = np.linalg.svd(base) #SVD
print(type(span))
print(span.shape)
print(s.shape)
print(vh.shape)
#print(span)
span=span[:,1:12] #i just assumed r, should do better
print(span)
span_T=span.transpose()
#get Projection
P=np.dot(span,span_T)
maxi=0
error=np.zeros(n_frames)
for i in range(n_frames-1):
    proj_f=np.dot(P,data[:,i])
    proj_f=proj_f/np.linalg.norm(proj_f)
    data_vector=data[:,i]/np.linalg.norm(data[:,i])
    #get error
    error_vector=np.subtract(data_vector,proj_f)
    error[i]=np.linalg.norm(error_vector)
    if error[i]>maxi:
        maxi=error[i]
histo=np.zeros(100)
#put in histogram
for i in range(n_frames-1):
    error[i]=error[i]/maxi
    ind=int(error[i]//0.01)
    histo[ind]+=1
#print(histo)
plt.hist(histo, bins='auto')  # arguments are passed to np.histogram
#plt.yscale('log')
plt.show()
error_ind=np.argsort(error) #index of errors by crescent order 

#get vals from video, wil be usefull to print skelectons
width=vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height=vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

#these are just the frame index
vid_skel_ind=skel[0,:]

#prints
'''for i in range(100):
    #choose some frame
    vid.set(cv2.CAP_PROP_POS_FRAMES,i+2944)
    res,frame=vid.read()
    skel_frame=np.where(vid_skel_ind==i+2943)
    #print skelectons on frame (might be more then 1 skelecton)
    for j in skel_frame[0]:
        print_skel(j,skel,frame,width,height)       
    cv2.imshow('Skelecton',frame)
    #cv2.waitKey(0)
    if cv2.waitKey(0) == ord('q'): #doesent work,need to refine later
        break
#stop functions
vid.release()
cv2.destroyAllWindows()'''


