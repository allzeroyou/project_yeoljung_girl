'''
파일 기능: 이미지를 머신러닝을 위한 pickle 파일로 변환
작성자 : 유다영
작성일 : 2022.05.26
기타: 인터넷에서 긁어옴, 코드를 우리 플젝에 맞게 수정 및 코드 이해 필요함, 아직 기능 작동 X
'''

from PIL import Image
from numpy import genfromtxt
import gzip
import _pickle
from glob import glob
import numpy as np
import pandas as pd
import imageio

img=imageio.imread("handwriting_number/1_0.png", pilmode='grayscale')
print(img.shape)
print(img)

# 아래 코드들은 수정해야 함
# df = pd.read_csv("train.csv", names = ["class"])
# np.array(df["class"]).shape

# def dir_to_dataset(glob_files, loc_train_labels=""):
#     print("Gonna process:\n\t %s" % glob_files)
#     dataset = []
#     for file_count, file_name in enumerate(sorted(glob(glob_files), key=len)):
#         img = imageio.imread(file_name, pilmode='RGB')
#         # print(img.shape)
#         # pixels = [f[0] for f in list(img.getdata())]
#         dataset.append(img)
#         if file_count % 1000 == 0:
#             print("\t %s files processed" % file_count)
#     # outfile = glob_files+"out"
#     # np.save(outfile, dataset)
#     if len(loc_train_labels) > 0:
#         df = pd.read_csv(loc_train_labels, names=["class"])
#         return np.array(dataset), np.array(df["class"])
#     else:
#         return np.array(dataset)


# Data1, y1 = dir_to_dataset("train/*.png", "train.csv")
# Data2, y2 = dir_to_dataset("valid/*.png", "valid.csv")
# Data3, y3 = dir_to_dataset("test/*.png", "test.csv")

# # Data and labels are read
# train_num = 2758
# valid_num = 844
# test_num = 420

# train_set_x = Data1[:train_num]
# train_set_y = y1[1:train_num + 1]
# val_set_x = Data2[:valid_num]
# val_set_y = y2[1:valid_num + 1]
# test_set_x = Data3[:test_num]
# test_set_y = y3[1:test_num + 1]


# # Divided dataset into 3 parts. I had 7717 images for training, 1653 images for validation and 1654 images for testing

# train_set = train_set_x, train_set_y
# val_set = val_set_x, val_set_y
# test_set = test_set_x, test_set_y

# dataset = [train_set, val_set, test_set]

# f = gzip.open('dy.pkl.gz','wb')
# _pickle.dump(dataset, f, protocol=2)
# f.close()

# len(val_set[0])
