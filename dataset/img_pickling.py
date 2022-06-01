'''
파일 기능: 이미지를 머신러닝을 위한 pickle 파일로 변환하기 전 파일
작성자 : 유다영
작성일 : 2022.05.26~31

변환된 파일명: dy.pkl.gz

기타: 6월 1일 회의 후
55번째 줄: 훈련, 검증, 시험데이터 비율 조정 및 변경 필요
'''

from PIL import Image
from numpy import genfromtxt
import gzip
import _pickle
from glob import glob
import numpy as np
import pandas as pd
import imageio

img=imageio.imread("handwriting_number/train/1_0.png", pilmode='RGB')
print(img.shape)
print(img)

# 아래 코드들은 우리 플젝에 맞춰 아주 조금 수정함(매우 좋은 코드였다)
df = pd.read_csv("train.csv", names = ["class"])
np.array(df["class"]).shape

def dir_to_dataset(glob_files, loc_train_labels=""):
    print("Gonna process:\n\t %s" % glob_files)
    dataset_dy = []
    for file_count, file_name in enumerate(sorted(glob(glob_files), key=len)):
        img = imageio.imread(file_name, pilmode='RGB')
        # print(img.shape)
        # pixels = [f[0] for f in list(img.getdata())]
        dataset_dy.append(img)
        if file_count % 1000 == 0:
            print("\t %s files processed" % file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)
    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels, names=["class"])
        return np.array(dataset_dy), np.array(df["class"])
    else:
        return np.array(dataset_dy)


Data1, y1 = dir_to_dataset("train/*.png", "train.csv")
Data2, y2 = dir_to_dataset("valid/*.png", "valid.csv")
Data3, y3 = dir_to_dataset("test/*.png", "test.csv")

# Data and labels are read
# 6월 1일 회의 후 훈련, 검증, 시험데이터 비율 조정 및 변경 예정
# 임시적으로 데이터셋 돌려보기 위해, train=dy, valid=hi, test=sj 했음

train_num = 140
valid_num = 100
test_num = 130

train_set_x = Data1[:train_num]
train_set_y = y1[1:train_num + 1]
val_set_x = Data2[:valid_num]
val_set_y = y2[1:valid_num + 1]
test_set_x = Data3[:test_num]
test_set_y = y3[1:test_num + 1]


# Divided dataset into 3 parts.
# I had 140 images for training, 100 images for validation and 130 images for testing

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('dy.pkl.gz', 'wb')
_pickle.dump(dataset, f, protocol=2)
f.close()

len(val_set[0])
