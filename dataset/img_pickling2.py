'''
파일 기능: 이미지를 머신러닝을 위한 pickle 파일로 변환
작성일 : 2022.05.31
기타:
mnist.py와 같은 기능을 해줌
최종 dy.pkl로 변경 후 mnist.pkl과 합쳐서 데이터셋 돌려보는 방법이 뭐지?
파이썬 코드 찾아보고

출처: https://www.notion.so/CNN-54af42a0bfa441ee90dc644416b78df6
'''

import _pickle, gzip, urllib.request, json
import numpy as np
from keras.utils import np_utils # from keras import utils as np_utils
from tensorflow.keras.utils import to_categorical

with gzip.open('dy.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = _pickle.load(f, encoding='utf-8')

(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = train_set, valid_set, test_set

train_images = train_images.astype('float32')
valid_images = valid_images.astype('float32')
test_images = test_images.astype('float32')

# normalizing the data to help with the training
train_images /= 255
valid_images /= 255
test_images /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 3

print("Shape before one-hot encoding: ", train_labels.shape)
train_labels = np_utils.to_categorical(train_labels, n_classes)
valid_labels = np_utils.to_categorical(valid_labels, n_classes)
test_labels = np_utils.to_categorical(test_labels, n_classes)
print("Shape after one-hot encoding: ", train_labels.shape)

from sklearn.preprocessing import LabelEncoder


def encoding_to_int(labels):
    e = LabelEncoder()
    e.fit(labels)
    return e.transform(labels)


n_classes = 3
print("Shape before one-hot encoding: ", train_labels.shape)
train_labels = np_utils.to_categorical(encoding_to_int(train_labels), n_classes)
valid_labels = np_utils.to_categorical(encoding_to_int(valid_labels), n_classes)
test_labels = np_utils.to_categorical(encoding_to_int(test_labels), n_classes)
print("Shape after one-hot encoding: ", train_labels.shape)
