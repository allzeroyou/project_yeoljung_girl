'''
파일 기능: 열정걸스만의 신경망
작성자 : 유다영
작성일 : 2022.05.26~31
기타:
- 텐서플로의 케라스 이용
- MNIST 데이터셋으로 이미지 정확도 99.5% 도출
- log message
1. Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
: https://developer.nvidia.com/cudnn에서 CUDA 버전에 맞는 것을 다운받고, 압축을 풀어서 CUDA가 설치된 폴더에 넣으면 해결됩니다.(https://evandde.github.io/nvcuda-not-found/)
2. None of the MLIR Optimization Passes are enabled (registered 2)
: MLIR은 Tensorflow 로직을 구현하고 최적화하기 위한 또 다른 솔루션으로 사용되고 있음. 이 정보 메시지는 MLIR이 사용되지 않았다는 내용입니다.
'''


# 1. MNIST 데이터 다운로드 및 확인
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 구조 확인
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print("Y[0] : ", y_train[0])
plt.imshow(x_train[0], cmap=plt.cm.gray_r, interpolation="nearest")


# 2. 합성곱 신경망(CNN, 컨볼루션 뉴럴 네트워크)
# 파라미터의 영향을 관찰하기 위해 random에 seed 부여(랜덤값 고정)
tf.random.set_seed(2022)

# Normalizing data
x_train, x_test = x_train / 255.0, x_test / 255.0

# (60000, 28, 28) => (60000, 28, 28, 1)로 reshape
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, input_shape=(28,28,1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
    # tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'), # 파라미터 수가 약 500개 대라 줄이기 위해 한 줄 주석처리
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=8, validation_data=(x_test, y_test))
# 에폭 또한 10 => 8개로 학습 시간을 줄임(CPU 사용량 감소를 위함)
result = model.evaluate(x_test, y_test)
print("합성곱 신경망을 이용한 이미지 정확도(%): : ", result[1]*100)