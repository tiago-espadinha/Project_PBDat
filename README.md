# Project_PBDat_G14

Manuel Palo, ist93120
Daniel Paulo, ist96173
Tiago Simoes, ist96329

## !!Bug fixes!!

Fixed outlier_detection() function to include a lower bound check (lines 122-123), minimal impact in the results.

## Code Description

The code provided in this repository is used to analyse the data collected from a cycling tour.

The data used are the following:

    - Skeleton data of the cyclist
    - Features data of the cyclist   

The incomplete skeleton data is reconstructed iteratively comparing it to the complete skeleton data.

The reconstructed skeleton data goes through dimensionality reduction with PCA to facilitate the outlier detection and clustering that follow

The features data also goes through dimensionality reduction with PCA to facilitate the outlier detection and clustering

The results of the outlier detection and clustering are plotted and saved in the `plots` folder

A new matrix is created from merging the data from the skeleton and the features

This new matrix is once again filtered for outliers and clustered being this results plotted and saved on the frames themselves to facilitate the visualisation of the results

## Dependencies
The code requires the following dependencies:

    - cv2 (OpenCV)
    - matplotlib
    - numpy
    - pandas
    - scipy
    - sklearn
    Please make sure to install these dependencies before running the code.

## Usage
To use the functions provided in this code, follow these steps:

    Check if all paths are correct 
        (search for 'path_check' to find any path used in the code)

    The necessary data should be added to the `data` folder:
        - 'girosmallveryslow2.mp4' is the video
        - 'girosmallveryslow2_openpose.mat' is the incomplete skeleton data
        - 'girosmallveryslow2_openpose_complete.mat' is the complete skeleton data
        - 'girosmallveryslow2.mp4_features.mat' is the features data

    Check flags and change them to the desired state 
        (search for 'flag_check' to find all flags used in the code)

    All the frames analysed are saved in the `frames` folder
        Warning: analysing all the frames and saving them can take a long time and will take a lot of space 
        (3.5GB approx.)
        if you don't want to save the frames, change the flag 'save_flag' to False
        alternatively, a smaller number of frames can be analysed by changing the flag 'subset' to True, and tweaking the 'frame_start' and 'frame_count' variables

    



