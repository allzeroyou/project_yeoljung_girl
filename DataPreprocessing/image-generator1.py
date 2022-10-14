# image generator 본 파일! 다른 값은 제가 임의로 넣어본것입니다.(예 : steps_per_epoch)


from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from tensorflow.keras.utils import img_to_array
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


trainDataGen = ImageDataGenerator(rescale=1./255,
                                 rotation_range = 10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 fill_mode='nearest'
                                 )


trainGenSet = trainDataGen.flow_from_directory(
    ' ' + 'train', #train 폴더의 위치 예) ./soojin_data/
    batch_size=64,
    target_size=(28,28),
    class_mode='categorical'
)


testDataGen = ImageDataGenerator(rescale=1./255)

testGenSet = testDataGen.flow_from_directory(
    ' ' + 'test',  #test 폴더의 위치 예) ./soojin_test/
    target_size=(28,28),
    batch_size=64,
    class_mode='categorical'
)

valDataGen = ImageDataGenerator(rescale=1./255,
                                 rotation_range = 15,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.1,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 fill_mode='nearest')

valGenSet = valDataGen.flow_from_directory(
    '.' + 'train', #train 폴더의 위치 예) ./soojin_data/
    target_size=(28,28),
    batch_size=64,
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(28,28,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax')) # 출력층의 갯수를 의미하는것으로 생각! 구글링해보기!!

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# fig_generator
model.fit_generator(
    trainGenSet,
    steps_per_epoch=10,  #임의로 변경함
    epochs=10,
    validation_data=valGenSet,
    validation_steps=10,
)

scores = model.evaluate_generator(testGenSet)
print(scores)






