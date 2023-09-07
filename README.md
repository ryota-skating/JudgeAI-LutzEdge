# Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs

Ryota Tanaka, Tomohiro Suzuki, Kazuya Takeda, Keisuke Fujii, Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs, 6th International ACM Workshop on Multimedia Content Analysis in Sports at ACM Multimedia 2023

# Introduction

This is the official code for "Automatic Edge Error Judgment in Figure Skating Using 3D Pose Estimation from a Monocular Camera and IMUs".

![overview_1](https://github.com/ryota-takedalab/JudgeAI-LutzEdge/assets/102862947/7c062b99-4ada-460b-82de-4d0e7a07c979)

# Datasets

You can download the IMU dataset as CSV files from `IMU_data/dataset` and video data from [Google Drive](https://drive.google.com/drive/folders/1WzERNs04uo_5xjybfKcXYOC9v8KL6Hk2?usp=drive_link).
Video data are pre-processed so that only the skaters are cut out from the bounding box, and the timing of the de-ice is aligned.

# Usage
You can validate our paper's data using the following code.

`python main.py training.csv`

By default, The model is trained using preprocessed data of 17 joint position coordinates of the whole body and the left skate pose angle, each at 60 fps.

The joint position coordinates and the left skate pose angle can each be downsampled to 12 fps using the option `--pos_fps 12` and `--rot fps 12`.

If you want to use only either the joint position coordinates or the left skate pose angle, options `--no_pos` and `--no_rot` can be used to reduce the unnecessary features.

In the random validation, you can change the number of trials using the option `-t (int)` or `--trials (int)`.

If you want to use keypoints only on the lower half of the body, not the whole body, you can use the option `--lower`.

You can save the validation result as a graph using the option `--plot`

# Author

Ryota Tanaka - tanaka.ryota@g.sp.m.is.nagoya-u.ac.jp
