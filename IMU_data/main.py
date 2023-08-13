import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import argparse
import itertools

# lower body (7keys)
L_KEY = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot"]

# whole body (17keys)
W_KEY = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot",
         "Spine", "Neck", "Neck1", "Head", "RightArm", "RightForeArm", "RightHand",
         "LeftArm", "LeftForeArm", "LeftHand"]

N_SKATER = 6
N_FRAME = 20


def feature_extraction(path_list, event_list, pos_fps, rot_fps, pos_key, rot_key):
    features = []
    use_pos_key = []
    use_rot_key = []
    for pk in pos_key:
        use_pos_key.append(pk + ".x")
        use_pos_key.append(pk + ".y")
        use_pos_key.append(pk + ".z")
    for rk in rot_key:
        use_rot_key.append(rk + ".z")

    print("\nPreprocessing dataset...")
    for path, event in tqdm(zip(path_list, event_list), leave=False):
        path_pos = path + "_pos.csv"
        path_rot = path + "_rot.csv"
        pos = pd.read_csv(path_pos, usecols=use_pos_key)
        rot = pd.read_csv(path_rot, usecols=use_rot_key)
        pos = pos.iloc[event - N_FRAME: event + N_FRAME, ]
        rot = rot.iloc[event - N_FRAME: event + N_FRAME, ]
        pos_array = pos.to_numpy()
        rot_array = rot.to_numpy()
        # shapeが(フレーム数, キーポイント数, 3(xyz座標))になるようreshapeする。
        pos_array = pos_array.reshape([len(pos_array), len(pos_key), 3])
        # pos: When dropping fps to 1/5
        if pos_fps == 12:
            pos_array = pos_array[::5]
        # rot: When dropping fps to 1/5
        if rot_fps == 12:
            rot_array = rot_array[::5]

        for i in range(len(pos_array)):
            for j in range(len(pos_key)):
                pos_array[i][j][1], pos_array[i][j][2] = pos_array[i][j][2], pos_array[i][j][1]

        # std (x,y,z) by Hip and foots
        for i in range(len(pos_array)):
            x_origin = pos_array[i][0][0]
            y_origin = pos_array[i][0][1]
            z_origin_r = pos_array[i][3][2]
            z_origin_l = pos_array[i][6][2]
            for j in range(int(len(pos_key))):
                pos_array[i][j][0] -= x_origin
                pos_array[i][j][1] -= y_origin
                if pos_array[i][3][2] < pos_array[i][6][2]:
                    pos_array[i][j][2] -= z_origin_r
                if pos_array[i][3][2] > pos_array[i][6][2]:
                    pos_array[i][j][2] -= z_origin_l

        for i in range(len(rot_array)):
            if rot_array[i] > 100:
                rot_array[i] = rot_array[i] - 180
            if rot_array[i] < -100:
                rot_array[i] = rot_array[i] + 180
        keypoint_1d = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(pos_array))))
        rot_array_1d = np.reshape(rot_array, -1)
        feature = np.array(keypoint_1d + rot_array_1d.tolist())
        features.append(feature)
    features_array = np.array(features)
    print("Preprocessing completed!!")

    return features_array


def plot_crossvalcoef(coef, pos_key, title, filename):
    # coef.shape = (n_keypoint, n_frame, n_skater)
    sns.set_style("ticks")
    sns.set_palette(["black"])
    y = np.zeros(shape=(coef.shape[0], coef.shape[2]))
    for n in range(coef.shape[2]):
        for k in range(coef.shape[0]):
            y[k][n] = coef[k, :, n].mean()

    y_df = pd.DataFrame(y, index=pos_key)
    y_df = y_df.T

    # sort by highest to lowest mean
    sorted_indices = np.argsort(- y_df.mean(axis=0))
    y_df = y_df[y_df.columns[sorted_indices]]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(
        data=y_df,
        showfliers=False,
        ax=ax,
        color="white",
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        medianprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    sns.stripplot(data=y_df, jitter=True, marker="o", color="black", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Importance")
    ax.set_ylim(0, 1.0)
    plt.title(title, fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig(filename)


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="python main.py training.csv [--pos_fps FPS(int)] [--rot_fps FPS(int)] [--no_pos] [--no_rot] [--lower] [--plot]",
        add_help=True,
    )
    parser.add_argument("train", help="training(.csv): filename of training data")
    parser.add_argument("--pos_fps", help="fps(int): FPS of the traning data (position)", type=int, default=60)
    parser.add_argument("--rot_fps", help="fps(int): FPS of the traning data (rotation)", type=int, default=60)
    parser.add_argument("--no_pos", help="(Option): Do not use psition data.", action="store_true")
    parser.add_argument("--no_rot", help="(Option): Do not use rotation data.", action="store_true")
    parser.add_argument("--lower", help="(Option): Use keypoints only on the lower half of the body, not the whole body.", action="store_true")
    parser.add_argument("--plot", help="(Option): Calculate lg-coef and plot them.", action="store_true")

    # analyze parser
    args = parser.parse_args()

    # load training data
    training = pd.read_csv(args.train)
    path_list = training["path"].to_numpy()
    event_list = training["event"].to_numpy()
    label_list = training["label"].to_numpy()

    # set pos keypoint
    if args.no_pos:
        pos_key = []
    else:
        if args.lower:
            pos_key = L_KEY
        else:
            pos_key = W_KEY

    n_key = len(pos_key)

    # set rot keypoint
    if args.no_rot:
        rot_key = []
    else:
        rot_key = ["LeftFoot"]

    # extract features of training data
    X = feature_extraction(path_list,
                           event_list,
                           pos_fps=args.pos_fps,
                           rot_fps=args.rot_fps,
                           pos_key=pos_key,
                           rot_key=rot_key,)
    y = np.array(label_list)

    # standardization
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    # number of frames
    fps = args.pos_fps
    n_frame = int(fps * N_FRAME / 30)

    # Logistic Regression
    lr = LogisticRegression(max_iter=10000)

    # player specific cross validation
    group_kfold = GroupKFold(n_splits=N_SKATER)

    # skaterA(29 data) : skaterB(37 data) : skaterC(36 data) :
    # skaterE(50 data) : skaterF(30 data) : skaterG(50 data)

    groups = []
    groups.extend([1] * 29)
    groups.extend([2] * 37)
    groups.extend([3] * 36)
    groups.extend([4] * 50)
    groups.extend([5] * 30)
    groups.extend([6] * 50)

    # cross validation
    score_funcs = ["accuracy", "f1"]
    scores = cross_validate(lr, X_std, y, groups=groups, scoring=score_funcs, cv=group_kfold, return_estimator=True)
    y_pred = cross_val_predict(lr, X_std, y, groups=groups, cv=group_kfold)
    # scores in each split
    print("\n--[Cross Validation Results]-------------------------------")
    print("Acc:", scores["test_accuracy"])
    print("F-Measure:", scores["test_f1"])
    print("Average Acc:", format(scores["test_accuracy"].mean() * 100, '.2f'), "±", format(scores["test_accuracy"].std() * 100, '.2f'), "%")
    print("Average F-Measure:", format(scores["test_f1"].mean() * 100, '.2f'), "±", format(scores["test_f1"].std() * 100, '.2f'), "%")

    # plot regression coefficients as boxplot
    if args.plot & (args.no_rot is False):
        crossval_coef = np.zeros(shape=(len(pos_key), n_frame, N_SKATER))
        for model, n in zip(scores["estimator"], range(N_SKATER)):
            regression_coef = np.abs(model.coef_[0])  # shape = (3*17*40, )
            x_n, y_n, z_n = 0, 1, 2
            for frame in range(n_frame):
                for k in range(n_key):
                    crossval_coef[k][frame][n] = regression_coef[x_n] + regression_coef[y_n] + regression_coef[z_n]
                    x_n += 3
                    y_n += 3
                    z_n += 3
        plot_crossvalcoef(
            crossval_coef,
            title="Joint pos. 12fps (PN3)",
            filename="crossval_coef_PN3_12fps_6skaters.png",
        )

    print_label = False
    if print_label:
        print("\ntest labels")
        print(y[[i == 1 for i in groups]])
        print(y[[i == 2 for i in groups]])
        print(y[[i == 3 for i in groups]])
        print(y[[i == 4 for i in groups]])
        print(y[[i == 5 for i in groups]])
        print(y[[i == 6 for i in groups]])
        print("predicted labels")
        print(y_pred[[i == 1 for i in groups]])
        print(y_pred[[i == 2 for i in groups]])
        print(y_pred[[i == 3 for i in groups]])
        print(y_pred[[i == 4 for i in groups]])
        print(y_pred[[i == 5 for i in groups]])
        print(y_pred[[i == 6 for i in groups]])


if __name__ == "__main__":
    main()
