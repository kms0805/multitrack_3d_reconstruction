# 3D Reconstruction of Multi-Objects' Trajectories
propose a pipeline that reconstructs trajectories of seagulls in three dimensions 
space using only videos about 50 to 100 seagulls flying on the beach taken at two viewpoints.

<img src = "src\overview.png">

## Results

<video src="result\3d.mp4"/>
<video src="result\tracking2deach.mp4"/>

## Implemention

    pip install -r requirements.txt

### centroid detection

    python seagull_detection.py
  
### calculate F matrix

    python calculateF.py
  
### matching

    python trajctory_matching.py

### 결과 동영상 저장하기

    python visualize.py --three_d --two_d
