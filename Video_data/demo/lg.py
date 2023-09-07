import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import argparse
import pickle
import seaborn as sns


# whole body 17keys
KEY = ["Hips", "R_UpLeg", "R_Leg", "R_Foot", "L_UpLeg", "L_Leg", "L_Foot", "Spine",
       "Neck", "Neck1", "Head", "R_Arm", "R_ForeArm", "R_Hand", "L_Arm", "L_ForeArm", "L_Hand"]
# lower body 7keys
NOT_KEY = ["Hips", "R_UpLeg", "R_Leg", "R_Foot", "L_UpLeg", "L_Leg", "L_Foot"]
# number of keypoints
N_KEY = len(KEY)
# number of skaters
N_SKATER = 6


def feature_extraction(path_list, fps=240, n_key=N_KEY):
    step = int(240 / fps)
    keypoints = []
    print("\nPreprocessing dataset...")
    for path in tqdm(path_list, leave=False):
        keypoint = list(np.load(path)["reconstruction"])
        keypoint = keypoint[::step]
        if n_key == 7:
            # 下半身のkeypoint、7点のみ使用の場合
            for f in range(len(keypoint)):
                keypoint[f] = keypoint[f][0:7]
        keypoint_1d = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(keypoint))))
        keypoints.append(np.array(keypoint_1d))
    keypoints_array = np.array(keypoints)
    print("Preprocessing completed!!")
    return keypoints_array


def plot(all_coef, title, filename):
    # plot regression coefficients with error bars
    y = np.zeros(shape=(N_KEY,))
    e = np.zeros(shape=(N_KEY,))
    for k in range(len(y)):
        y[k] = all_coef[k].mean()
        e[k] = all_coef[k].std()
    y_sorted = np.sort(y)[::-1]
    x = np.array(KEY)[np.argsort(y)[::-1]]
    x_position = np.arange(len(x))
    fig, ax = plt.subplots()
    ax.bar(x_position, y_sorted, tick_label=x, color="darkgray")
    ax.set_ylabel("Importance")
    ax.set_ylim(0, 1.0)
    fig.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=90)
    plt.title(title, fontsize=16, y=1.03)
    plt.tight_layout()
    plt.savefig(filename)


def plot_crossvalcoef(coef, title, filename):
    # coef.shape = (n_keypoint, n_frame, n_skater)
    # sns.set()
    sns.set_style("ticks")
    sns.set_palette(["black"])
    y = np.zeros(shape=(coef.shape[0], coef.shape[2]))
    for n in range(coef.shape[2]):
        for k in range(coef.shape[0]):
            y[k][n] = coef[k, :, n].mean()

    y_df = pd.DataFrame(y, index=KEY)
    y_df = y_df.T

    # 平均値が高い順に並べ替える
    sorted_indices = np.argsort(-y_df.mean(axis=0))
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


def regress_and_plot(X_train, X_test, y_train, y_test, skater="A", key=17, fps=12):
    # set n_frame (12fps -> 8frame, 60fps -> 40frame)
    n_frame = int(fps * (160 / 240))
    # Logistic Regression
    lr = LogisticRegression(max_iter=10000)
    # learn Models without skaterA
    lr.fit(X_train, y_train)
    # evaluate models
    y_predict = lr.predict(X_test)
    acc = accuracy_score(y_test, y_predict) * 100
    f1 = f1_score(y_test, y_predict) * 100
    print("---------------------------------------------------")
    print(f"cross-validation / test data: skater{skater}")
    print(f"Acc : {format(acc, '.2f')}%")
    print(f"F-Measure : {format(f1, '.2f')}%")
    print(f"skater{skater} (test)     : {y_test}")
    print(f"skater{skater} (predicted): {y_predict}")
    # calculate regression coef
    regression_coef = np.abs(lr.coef_[0])  # shape = (3*17*40, )
    coef = np.zeros(shape=(N_KEY, n_frame))
    x_n, y_n, z_n = 0, 1, 2
    for frame in range(n_frame):
        for k in range(N_KEY):
            coef[k][frame] = regression_coef[x_n] + regression_coef[y_n] + regression_coef[z_n]
            x_n += 3
            y_n += 3
            z_n += 3
    # plot regression coef
    plot(
        coef,
        title=f"test data: skater{skater}",
        filename=f"result/coef_test{skater}_STpose3D_{fps}fps.png",
    )

    return coef


def main():
    parser = argparse.ArgumentParser(
        prog="lg.py",
        usage="python lg.py training.csv [-f FPS(int)] [-t TRIAL(int)] [--plot]",
        add_help=True,
    )
    parser.add_argument("training", help="training(.csv): filename of training data")
    parser.add_argument("-f", "--fps", help="fps(int): FPS of the traning data", type=int, default=60)
    parser.add_argument("--plot", help="(Option): Calculate lg-coef and plot them.", action="store_true")

    # analyze parser
    args = parser.parse_args()
    # set fps
    fps = args.fps
    # set n_frame (12fps -> 8frame, 60fps -> 40frame)
    n_frame = int(fps * (160 / 240))
    # load training data
    training = pd.read_csv(args.training)
    path_list = training["path"].to_numpy()
    label_list = training["label"].to_numpy()

    # extract features of training data
    X = feature_extraction(path_list, fps=fps)
    y = np.array(label_list)

    # standardization
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    pickle.dump(sc, open(f'./demo/lr_data/scaler_{fps}fps.sav', 'wb'))

    # Logistic Regression
    lr = LogisticRegression(max_iter=10000)

    print("\nCalicurating Logistic Regression...")
    # 選手別クロスバリデーション
    group_kfold = GroupKFold(n_splits=N_SKATER)

    # skaterA(29データ) : skaterB(37データ) : skaterC(36データ)
    # skaterD(48データ) : skaterE(29データ) : skaterF(50データ)
    groups = []
    groups.extend([1] * 29)
    groups.extend([2] * 37)
    groups.extend([3] * 36)
    groups.extend([4] * 48)
    groups.extend([5] * 29)
    groups.extend([6] * 50)

    # cross validation
    score_funcs = ["accuracy", "f1"]
    scores = cross_validate(lr, X_std, y, groups=groups, scoring=score_funcs, cv=group_kfold, return_estimator=True)
    y_pred = cross_val_predict(lr, X_std, y, groups=groups, cv=group_kfold)
    # scores in each split
    print("\n--[Cross Validation]-------------------------------")
    print("Acc:", scores["test_accuracy"])
    print("F-Measure:", scores["test_f1"])
    print(
        "Average Acc:",
        format(scores["test_accuracy"].mean() * 100, ".2f"),
        "±",
        format(scores["test_accuracy"].std() * 100, ".2f"),
        "%",
    )
    print(
        "Average F-Measure:",
        format(scores["test_f1"].mean() * 100, ".2f"),
        "±",
        format(scores["test_f1"].std() * 100, ".2f"),
        "%",
    )
    # print regression coef
    crossval_coef = np.zeros(shape=(len(KEY), n_frame, N_SKATER))
    for model, n in zip(scores["estimator"], range(N_SKATER)):
        regression_coef = np.abs(model.coef_[0])  # shape = (3*17*40, )
        x_n, y_n, z_n = 0, 1, 2
        for frame in range(n_frame):
            for k in range(N_KEY):
                crossval_coef[k][frame][n] = regression_coef[x_n] + regression_coef[y_n] + regression_coef[z_n]
                x_n += 3
                y_n += 3
                z_n += 3
    plot_crossvalcoef(
        crossval_coef,
        title=f"Joint pos. {fps}fps (ST-Pose3D)",
        filename=f"result/crossval_coef_STpose3D_{fps}fps.png",
    )

    # split into training and validation data
    X_std_A = X_std[[i == 1 for i in groups]]
    X_std_B = X_std[[i == 2 for i in groups]]
    X_std_C = X_std[[i == 3 for i in groups]]
    X_std_D = X_std[[i == 4 for i in groups]]
    X_std_E = X_std[[i == 5 for i in groups]]
    X_std_F = X_std[[i == 6 for i in groups]]
    y_A = y[[i == 1 for i in groups]]
    y_B = y[[i == 2 for i in groups]]
    y_C = y[[i == 3 for i in groups]]
    y_D = y[[i == 4 for i in groups]]
    y_E = y[[i == 5 for i in groups]]
    y_F = y[[i == 6 for i in groups]]
    X_std_woA = X_std[[i != 1 for i in groups]]
    X_std_woB = X_std[[i != 2 for i in groups]]
    X_std_woC = X_std[[i != 3 for i in groups]]
    X_std_woD = X_std[[i != 4 for i in groups]]
    X_std_woE = X_std[[i != 5 for i in groups]]
    X_std_woF = X_std[[i != 6 for i in groups]]
    y_woA = y[[i != 1 for i in groups]]
    y_woB = y[[i != 2 for i in groups]]
    y_woC = y[[i != 3 for i in groups]]
    y_woD = y[[i != 4 for i in groups]]
    y_woE = y[[i != 5 for i in groups]]
    y_woF = y[[i != 6 for i in groups]]

    # learn Models without skaterA
    regress_and_plot(X_std_woA, X_std_A, y_woA, y_A, skater="A", key=N_KEY, fps=fps)
    # learn Models without skaterB
    regress_and_plot(X_std_woB, X_std_B, y_woB, y_B, skater="B", key=N_KEY, fps=fps)
    # learn Models without skaterC
    regress_and_plot(X_std_woC, X_std_C, y_woC, y_C, skater="C", key=N_KEY, fps=fps)
    # learn Models without skaterD
    regress_and_plot(X_std_woD, X_std_D, y_woD, y_D, skater="D", key=N_KEY, fps=fps)
    # learn Models without skaterE
    regress_and_plot(X_std_woE, X_std_E, y_woE, y_E, skater="E", key=N_KEY, fps=fps)
    # learn Models without skaterF
    regress_and_plot(X_std_woF, X_std_F, y_woF, y_F, skater="F", key=N_KEY, fps=fps)
    
    # モデルを保存する
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_std, y)
    filename = f'./demo/lr_data/lr_model_{fps}fps.sav'
    pickle.dump(lr, open(filename, 'wb'))


if __name__ == "__main__":
    main()
