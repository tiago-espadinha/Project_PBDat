# PBDat Project G14

## Team Members

- Manuel Palo, ist93120
- Daniel Paulo, ist96173
- Tiago Simões, ist96329

## Project Description

This repository contains code for analyzing data collected from a cycling tour. The analysis pipeline includes:

1. **Skeleton Data Reconstruction**: Incomplete skeleton data is iteratively reconstructed by comparing it to complete skeleton data.
2. **Dimensionality Reduction**: Both skeleton and features data undergo PCA processing to facilitate outlier detection and clustering.
3. **Outlier Detection**: Identifies anomalous data points in both skeleton and features datasets.
4. **Clustering**: Groups similar data patterns after outlier removal.
5. **Feature Fusion**: Creates a new matrix by merging skeleton and features data.
6. **Visualization**: Results are plotted and saved in the `plots` folder, with optional frame-by-frame visualization.

## Requirements

- OpenCV (cv2)
- matplotlib
- numpy
- pandas
- scipy
- scikit-learn (sklearn)

### Data Requirements

The following data files must be placed in the `data` directory:

- `girosmallveryslow2.mp4` - Input video
- `girosmallveryslow2_openpose.mat` - Incomplete skeleton data
- `girosmallveryslow2_openpose_complete.mat` - Complete skeleton data
- `girosmallveryslow2.mp4_features.mat` - Extracted features data

## Installation

1. Install dependencies as specified above:

```bash
pip install opencv-python matplotlib numpy pandas scipy scikit-learn
```

2. Prepare data files as required
3. Configure any necessary settings

## Usage

### Configuration

- Search for `flag_check` in the code to find all configurable flags

### Running the Project

1. Place data files in the `data/` directory as specified above
2. Set configuration flags in the script
3. Execute the main script:

```bash
python PBDat_Giro_Proj.py
```

### Optional Flags/Parameters

- `save_flag`: Set to True to save analyzed frames (requires ~3.5GB storage), set to False to skip frame saving
- `subset_flag`: Set to True with `frame_start` and `frame_count` to analyze a subset of frames
- `print_plots`: Set to True to show plots of the data
- `print_frames`: Set to True to show frames of the video
- `skel_flag`: Set to True to print the skeleton on the frame (needs print_frames = True)

## Outputs

Describe what the project produces and where it is saved:

- Plots saved in `plots/` directory
- Analyzed frames saved in `frames/` directory (if `save_flag = True`)

## Notes

Search for `path_check` in the code to confirm all file paths are correct.
