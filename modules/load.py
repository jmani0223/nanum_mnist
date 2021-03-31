import os
import cv2
import numpy as np

def rename(path_dir):
    dir_list = os.listdir(path_dir)
    for dir in dir_list:
        dir_list_2 = os.listdir(os.path.join(path_dir, dir))
        print(dir_list_2)
        i = 1
        for name in dir_list_2:
            src = os.path.join(path_dir, dir, name)
            dst = str(i) + '.jpg'
            dst = os.path.join(path_dir, dir, dst)
            os.rename(src, dst)
            i += 1


def dataload(path_dir):
    X = []
    Y = []
    dir_list = os.listdir(path_dir)

    for dir in dir_list:
        dir_list_2 = os.listdir(path_dir + '/' + dir)
        for filename in dir_list_2:
            img = cv2.imread(path_dir + '/' + dir + '/' + filename)
            img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)  # 사이즈 변경
            X.append(img / 256)
            Y.append(int(dir))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def hangle_to_num(path_dir):
    dir_list = os.listdir(path_dir)
    for dir in dir_list:
        dir_list_2 = os.listdir(os.path.join(path_dir, dir))
        i = 1
        for name in dir_list_2:
            src = os.path.join(path_dir, dir, name)
            dst = str(i) + '.jpg'
            dst = os.path.join(path_dir, dir, dst)
            os.rename(src, dst)
            i += 1

