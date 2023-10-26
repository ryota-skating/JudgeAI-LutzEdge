# Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs

Ryota Tanaka, Tomohiro Suzuki, Kazuya Takeda, Keisuke Fujii, Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs, 6th International ACM Workshop on Multimedia Content Analysis in Sports at ACM Multimedia 2023

# Introduction

This is the official code for "Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs".

![overview_1](https://github.com/ryota-takedalab/JudgeAI-LutzEdge/assets/102862947/7c062b99-4ada-460b-82de-4d0e7a07c979)

# Datasets

You can download the IMU dataset as CSV files from `IMU_data/dataset` and video data from [Google Drive](https://drive.google.com/drive/folders/1WzERNs04uo_5xjybfKcXYOC9v8KL6Hk2?usp=drive_link).
Video data are pre-processed so that only the skaters are cut out from the bounding box, and the timing of the take-off is aligned.

# Usage
You can validate our paper's data using the following code.
Note: The GIF image in the example command execution below is played back at 20x speed.

<img src="https://github.com/ryota-takedalab/JudgeAI-LutzEdge/assets/102862947/b088c223-fbd9-45b7-83ca-f15b496a73c2" width="600">

## STEP1
Upload the video you want to judge for edge error `Video_data/demo/video`. However, the video is supposed to be at 240 fps with a fixed viewpoint.
For video file extensions, .mp4 and .mov are recommended.

## STEP2
Run the following code to the `Video_data/` directory.

`python demo/main.py --video sample_video.mov`

First, it detects who jumps in the video on a bbox basis.
Next, 2D pose estimation of the target person is performed based on the detected bbox.
Finally, 3D pose estimation is performed based on the estimated 2D pose to judge edge errors.
The result is displayed as either "EDGE ERROR" or "NOT EDGE ERROR," with the prediction accuracy displayed simultaneously.

# Author

Ryota Tanaka - tanaka.ryota@g.sp.m.is.nagoya-u.ac.jp
