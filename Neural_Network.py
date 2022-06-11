## femmefetalehaein의 TrainByCsv.py 수정 파일

import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical, np_utils
import matplotlib.pyplot as plt
from tensorflow import keras

# 교수님이 주신 테스트셋 적용
with open('./test_data/testdata1D.pkl', 'rb') as f:
    datasetL = pickle.load(f)
x_test, y_test = datasetL

# train 데이터셋을 csv로 변환한 것
train=pd.read_csv('./mnist_train.csv')


# 행단위로 적용된다.
x_train = train.drop('label', axis=1)
y_train = train['label'] #target

# 입력을 0~255에서 0~1로 정규화
x_train=x_train/255

# CNN 모델 입력층에 맞춰서 reshape하는 과정
x_train = np.array(x_train).reshape((-1, 28, 28, 1))
x_test = x_test.reshape(429, 28, 28, 1).astype('float32') / 255.0

# one-hot 인코딩 처리를 수행 , num_classes은 최종 출력 클래스의 크기
y_test = np_utils.to_categorical(y_test)
y_train = to_categorical(y_train, num_classes=10)

# 형태 확인
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

# 모델 구성
model = keras.Sequential([
keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)),  # 필터의 수 32개, 필터 크기 3 X 3, 활성화 함수 'relu', 입력 이미지 크기 28 * 28, 입력 이미지 채널 1( 흑백일 경우 채널1, 컬러일 경우 채널 3)
keras.layers.Conv2D(64, (3, 3), activation='relu'), #컨볼리션 레이어 : 필터 크기 3*3, 필터 수 64개, 활성화 함수 'relu'
keras.layers.MaxPooling2D(pool_size=(2, 2)), #맥스 풀링 레이어 : 풀 크기  2 * 2
keras.layers.Conv2D(32, (2, 2), activation='relu'), #컨볼리션 레이어 : 필터 크기 3*3, 필터 수 32개, 활성화 함수 'relu'
keras.layers.MaxPooling2D(pool_size=(2, 2)), #맥스 풀링 레이어 : 풀 크기  2 * 2
keras.layers.Flatten(), #플래튼 레이어
keras.layers.Dense(256, activation='relu'), #댄스 레이어 : 출력 뉴런수 256개, 활성화 함수: 'relu'
keras.layers.Dense(10, activation='softmax') #댄스 레이어: 출력 뉴런수 10개, 활성화 함수 'softmax'
    ]
)

# loss는 현재 가중치 세트를 평가하는데 사용한 손실 함수 , optimizer은 최적화 알고리즘, metrics는 평가 척도를 나타냄, 학습률은 0.0002
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 모델 학습시키기
hist = model.fit(x_train, y_train, batch_size= 10, epochs=10)

# 모델 저장
model.save('deep_model_final.h5')

# 행렬의 축을 지정, 분류한 결과를 확인
pred = np.argmax(model.predict(x_train), axis=-1)

# 모델 평가하기
result = model.evaluate(x_test, y_test, verbose=2)
print("최종 예측 성공률(%): ", result[1]*100)


# 결과 그래프 출력
plot_target = ['loss' , 'accuracy']
plt.figure(figsize=(12, 8))

for each in plot_target:
    plt.plot(hist.history[each], label = each)
plt.legend()
plt.grid()
plt.show()

# 몇개의 결과를 이미지로 확인
predicted_result = model.predict(x_train)
predicted_labels = np.argmax(predicted_result, axis=1)
test_labels = np.argmax(y_train, axis=1)
plt.figure(figsize = [12,12])


for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(x_train[i])
    tmp = "Label:" + str(test_labels[i]) + ", Prediction:" + str(predicted_labels[i])
    plt.title(tmp)
    print(y_train[i]) # 무슨 숫자인지 나타냄
plt.show()


