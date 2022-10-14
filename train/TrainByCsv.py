import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#이게뭐지?
import tensorflow
from keras.utils import to_categorical
import time
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import optimizers
from keras import metrics

start = time.time()

test=pd.read_csv('./mnist_test.csv')
train=pd.read_csv('./mnist_train.csv')

#라벨 분리

#행단위로 적용된다.
x_train = train.drop('label', axis=1)
y_train = train['label'] #target

x_test = test.drop('label', axis=1) #test
y_test = test['label']
#x_test = test.drop('label',axis=1)
#데이터표준화 (0-255값을 가지니까 픽셀마다)


x_train=x_train/255
x_test=x_test/255

#CNN 모델 입력층에 맞춰서 reshape하는 과정
x_train = np.array(x_train).reshape((-1, 28, 28, 1))
x_test = np.array(x_test).reshape((-1, 28, 28, 1))

#label= to_categorical(y_train, num_classes=10)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

####################여기가 무슨역할인지 모르겠다####
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
print(y_train.shape)
##################### 확인요망 ################

model = keras.Sequential([
keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)),  # 필터의 수 64개, 필터 크기 3 X 3, 활성화 함수 'relu', 입력 이미지 크기 28 * 28, 입력 이미지 채널 3( 흑백일 경우 채널1, 컬러일 경우 채널 3)
keras.layers.Conv2D(64, (3, 3), activation='relu'), #컨볼리션 레이어 : 필터 크기 3*3, 필터 수 64개, 활성화 함수 'relu'
keras.layers.MaxPooling2D(pool_size=(2, 2)), #맥스 풀링 레이어 : 풀 크기  2 * 2
keras.layers.Conv2D(32, (2, 2), activation='relu'), #컨볼리션 레이어 : 필터 크기 3*3, 필터 수 128개, 활성화 함수 'relu'
keras.layers.MaxPooling2D(pool_size=(2, 2)), #맥스 풀링 레이어 : 풀 크기  2 * 2
keras.layers.Flatten(), #플래튼 레이어
keras.layers.Dense(256, activation='relu'), #댄스 레이어 : 출력 뉴런수 256개, 활성화 함수: 'relu'
keras.layers.Dense(10, activation='softmax')
    ]
    #댄스 레이어: 출력 뉴런수 10개, 활성화 함수 'softmax'
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # 각 계층별 출력 데이터의 차원을 확인하실 수 있습니다.

#배치사이즈 및 에폭 설정 어떻게 할까?
hist = model.fit(x_train, y_train, batch_size= 10, epochs=10)  # 모델 학습

pred = np.argmax(model.predict(x_train), axis=-1)  # 분류한 결과를 확인하실 수 있습니다.

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
acc_ax.plot(hist.history['acc'],'b',label='train acc')
acc_ax.plot(hist.history['val_acc'],'g',label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()



