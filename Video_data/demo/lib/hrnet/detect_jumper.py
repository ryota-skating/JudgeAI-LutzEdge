from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds
from lib.hrnet.lib.models import pose_hrnet

cfg_dir = "demo/lib/hrnet/experiments/"
model_dir = "demo/lib/checkpoint/"

# Loading human detector model
from lib.yolov3.human_detector import load_model as yolo_model
from lib.yolov3.human_detector import yolo_human_det as yolo_det
from lib.sort.sort import Sort

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from PIL import Image
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", type=str, default=cfg_dir + "w48_384x288_adam_lr1e-3.yaml", help="experiment configure file name"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, default=None, help="Modify config options using the command-line"
    )
    parser.add_argument(
        "--modelDir", type=str, default=model_dir + "pose_hrnet_w48_384x288.pth", help="The model directory"
    )
    parser.add_argument("--det-dim", type=int, default=416, help="The input dimension of the detected image")
    parser.add_argument("--thred-score", type=float, default=0.30, help="The threshold of object Confidence")
    parser.add_argument("-a", "--animation", action="store_true", help="output animation")
    parser.add_argument("-np", "--num-person", type=int, default=1, help="The maximum number of estimated poses")
    parser.add_argument("-v", "--video", type=str, default="camera", help="input video file name")
    parser.add_argument("--gpu", type=str, default="0", help="input video")
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')

    return model


def get_locus(bboxs_y, filename="locus.png"):
    np.putmask(bboxs_y, bboxs_y == 0, np.nan)
    bboxs_y_grad = np.zeros(bboxs_y.shape, dtype=np.float32)
    # get locus gradient and plot locus
    plt.figure(figsize=(12, 8))
    for i, bbox_height in enumerate(bboxs_y):
        x = range(len(bbox_height))
        y = bbox_height
        y_smooth = savgol_filter(y, 175, 2)
        y_grad = np.gradient(y_smooth)
        plt.plot(x, y_grad, linewidth=2, label="ID: " + str(i + 1))
        bboxs_y_grad[i] = y_grad
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show
    plt.savefig(filename)

    return bboxs_y_grad


def detect_jumper(bboxs_y1_grad, bboxs_y2_grad, frame=100, threshold_1=0.8, threshold_2=0.08):
    """ジャンプしている人を検出し、その人のIDを返す.

    Args:
        bboxs_y_grad (np.ndarray): 各IDのbboxの位置の変化率データ.
        frame (int, optional): ジャンプの滞空フレーム数. Defaults to 80.
        threshold_1 (float, optional): bboxの上辺の位置の変化率の閾値。Defaults to 0.8.
        threshold_2 (float, optional): bboxの底辺の位置の変化率の閾値。Defaults to 0.08.

    Returns:
        jumper (list): ジャンプしている人のIDを格納したlist.
        jump_timing (list): ジャンプの最高点のタイミング
    """

    jumper = []
    jump_timing = []
    id = 1
    for bbox_y1_grad, bbox_y2_grad in zip(bboxs_y1_grad, bboxs_y2_grad):
        for i in range(int(len(bbox_y1_grad) - frame)):
            center = i + int(frame / 2)
            if (bbox_y1_grad[center - int(frame / 2)] < -threshold_1) and (
                bbox_y1_grad[center + int(frame / 2)] > threshold_1
            ):
                if (bbox_y2_grad[center - int(frame / 2)] < -threshold_2) and (
                    bbox_y2_grad[center + int(frame / 2)] > threshold_2
                ):
                    jumper.append(id)
                    jump_timing.append(
                        center
                        - int(frame / 2)
                        + np.argmin(np.abs(bbox_y1_grad[center - int(frame / 2) : center + int(frame / 2)]))
                    )
                    break
        id += 1
    if not jumper:
        threshold_1 -= 0.1
        if threshold_1 > 0:
            jumper, jump_timing = detect_jumper(bboxs_y1_grad, bboxs_y2_grad, frame, threshold_1=threshold_1)
        else:
            print("No jumper detected!!")
    if len(jumper) > 1:
        threshold_1 += 0.03
        jumper, jump_timing = detect_jumper(bboxs_y1_grad, bboxs_y2_grad, frame, threshold_1=threshold_1)

    return jumper, jump_timing


def draw_bboxs(img, people_track):
    # cv2は左上原点で描画することに注意！！
    for i in range(len(people_track)):
        # pt1はbbox左下、pt2はbbox右上
        cv2.rectangle(
            img,
            pt1=(int(people_track[i][1]), int(people_track[i][4])),
            pt2=(int(people_track[i][3]), int(people_track[i][2])),
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_4,
            shift=0,
        )

        cv2.putText(
            img,
            text="ID:" + str(int(people_track[i][0])),
            org=(int(people_track[i][1]), int(people_track[i][2]) - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4,
        )

    return img



# vis.pyのget_pose2D関数の中で、hrnet_pose関数という名前で使われてる。
def gen_video_kpts(video, det_dim=416, cut_frame=160):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    # 動画ファイルの読み込み
    cap = cv2.VideoCapture(video)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    # get()メソッドを用いて、読み込んだ動画の総フレーム数を取得
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    track_bboxs = []
    im_list = []
    print("\nDetecting jumper...")
    for _ in tqdm(range(video_length), leave=False):
        # retはフレームの画像が読み込めたかどうかを示すbool値
        # frameは画像の配列ndarrayのタプル
        # read()の度に1フレームずつ読み込んでいく
        ret, frame = cap.read()

        if not ret:
            continue

        bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

        bboxs_pre = []
        scores_pre = []
        if bboxs is None or not bboxs.any():
            print("No person detected!")
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using Sort to track people
        # [[ID, x1, y1, x2, y2], ...]の形でソート。IDは昇順
        people_track = people_sort.update(bboxs)[::-1, [4, 0, 1, 2, 3]]
        track_bboxs.append(people_track)

        # bboxを描画
        img_array = draw_bboxs(frame, people_track)
        img = Image.fromarray(img_array[:, :, ::-1])
        img = img.resize((int(img.width / 2), int(img.height / 2)))
        im_list.append(img)

    # bboxデータをGIFファイルで保存
    im_list[0].save(
        "out.gif", save_all=True, append_images=im_list
    )

    num_ID = 0
    for t_bboxs in track_bboxs:
        if num_ID < int(np.max(t_bboxs[:, 0])):
            num_ID = int(np.max(t_bboxs[:, 0]))
    bboxs_y1 = np.zeros((num_ID, video_length), dtype=np.float32)
    bboxs_y2 = np.zeros((num_ID, video_length), dtype=np.float32)
    bboxs_height = np.zeros((num_ID, video_length), dtype=np.float32)

    # 各フレームにおけるbboxの上側(頭)のy座標と、サイズ(高さ、身長)を抽出
    for t, t_bboxs in enumerate(track_bboxs):
        for id_box in t_bboxs:
            id = int(id_box[0]) - 1
            bboxs_y1[id][t] = id_box[2]
            bboxs_y2[id][t] = id_box[4]
            bboxs_height[id][t] = id_box[4] - id_box[2]

    # plot locus
    bboxs_y1_grad = get_locus(bboxs_y1, filename="y1_locus.png")
    bboxs_y2_grad = get_locus(bboxs_y2, filename="y2_locus.png")

    # ジャンプしている人のIDを求める
    jumper, jump_timing = detect_jumper(bboxs_y1_grad, bboxs_y2_grad, frame=100, threshold_1=0.6, threshold_2=0.1)

    # id = 2
    # jump = 461
    # start_frame = jump - cut_frame

    if len(jumper) != 0:
        print(f"jumper detected!! \nID: {jumper[0]}")

        # 求めたIDの人物の姿勢推定を行う
        id = jumper[0]
        jump = jump_timing[0]
        start_frame = jump - cut_frame
        if start_frame < 0:
            # start_frame = 0
            print("撮り始め遅いかも")
            sys.exit()

    else:
        print("ジャンプ検出できなかったっぽい")
        sys.exit()

    track_bboxs_ = []
    for track_bbox in tqdm(track_bboxs[start_frame : start_frame + cut_frame], leave=False):

        if len(np.where(track_bbox[:, :1] == id)[0]) != 0:
            where_id = np.where(track_bbox[:, :1] == id)[0][0]
            for bbox in track_bbox[where_id][1:].reshape(1, 4):
                bbox = [math.floor(i) for i in list(bbox)]
                # track_bboxs_ = [[x1, y1, x2, y2]]
                track_bboxs_.append(bbox)

    return start_frame, track_bboxs_
