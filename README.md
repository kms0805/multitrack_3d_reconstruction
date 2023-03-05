# 3D Reconstruction of Multi-Objects' Trajectories From Uncalibrated Multi-View Video

I conducted this project as part of my graduation requirements for the Electrical and Computer Engineering at Seoul National University.

## Overview
The project focuses on reconstructing the trajectories of seagulls in three-dimensional space from two different viewpoints using uncalibrated multi-view videos. The pipeline involves detecting the center of gravity of seagull objects, tracking them, and complementarily connecting the trajectories of the images from two viewpoints. The project employs techniques such as background subtraction, seagull detection using prior knowledge, density-based clustering, feature point extraction, and Graph-Cut RANSAC algorithm for estimating the optimal fundamental matrix. The trajectories of each individual are connected according to a matching algorithm by calculating the cost between the pieces of trajectory from different viewpoints. Finally, the cost is recalculated and the matching algorithm is executed to participate in the trajectory pieces that couldn’t participate in matching.

<img src = "src\overview.png">
<br>
<br>
<br>

## Results

<img src="src\3d.gif" width = 400>
<img src="src\tracking2deach.gif"/>


<br>
<br>
<br>

## Implemention

    pip install -r requirements.txt

### centroid detection

    python seagull_detection.py
  
### calculate F matrix

    python calculateF.py
  
### matching

    python trajctory_matching.py

### visualize

    python visualize.py --three_d --two_d
