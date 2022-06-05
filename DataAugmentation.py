''''
파일 기능: 이미지 어그멘테이션을 위한 파일
작성자 : 유다영
작성일 : 2022.06.02
기타: 회전, 뒤집기 등 훈련 데이터의 폭을 넓히기 위한 어그멘테이션 기능입니동
하지만 아직 손 봐야 합니다..
- 6월 5일 업데이트
코드봤지만 아직도 왜 1개 파일만 생기는지 모르겠다.
save 함수를 어떻게 해야 할 듯 한데, 파라미터가 많아서 손보기가 어렵다.
내일 다시 도전
할 수 있다!

- 손 봐야 할 것
train > 2의 폴더 전체를 돌렸는데, 출력결과는 _rotate.png 파일한개뿐?
여러 파일이 생기지 않는 이유?

출처: https://github.com/oryondark/hjkim/blob/master/DeepLearning_master/Augmentation_Tutorial/Data%20Augmentation.md
'''

# import modules for image preprocessing

import cv2 as cv
import matplotlib.pyplot as plt
import os, sys

# image parse in directory

root_dir_path = './dataset/handwriting_number/train/2'  # target images directory
root_dir = os.listdir(root_dir_path)

# search = '.png'
# for i, word in enumerate(root_dir):
#     if search in word:
#         print('>> modify: ' + word)
#         root_dir[i] = word.strip(search)
root_dir = sorted(root_dir)
print(root_dir)  # 폴더 내 파일명으로 오름차순 정렬


# Create Save method by openCV
# keyPath는 원본 이미지의 루트 경로
# file_name은 원본 이미지 파일 이름
# cv_img는 이미지의 전체 신호
# 비율은 스케일 값에 대한 것
def save(keyPath, file_name, cv_img, rate, type):
    if not os.path.isdir(keyPath):
        os.mkdir(keyPath)

    saved_name = os.path.join(keyPath, "{}{}.{}".format(file_name.split('.')[0], type, 'png'))
    print(saved_name)
    cv.imwrite(saved_name, cv_img)


# 1.  Scaling
def augmente(keyName, rate=None, if_scale=False):
    saved_dir = "./augmentation_images"
    keyPath = os.path.join(root_dir_path, keyName)  # keypath direct to root path
    keyPath_rep = keyPath.replace('\\', '/')
    keyPath_rep_list = []
    i=0
    while len(keyPath_rep_list) <= 9:
        for i in range(10):
            keyPath_rep_list.append(keyPath_rep[:39]+str(i)+'.png')
            print("keyPath: ", keyPath_rep_list[i])
            # keyPath_rep += 1
            print(len(keyPath_rep_list))
    datas = keyPath_rep_list
    data_total_num = len(datas)

    print("Overall data in {} Path :: {}".format(keyPath_rep, data_total_num))

    try:
        for data in datas:
            type = "_scale_"
            # data_path = os.path.join(keyPath_rep, data)
            img = cv.imread(keyPath_rep)
            shape = img.shape
            ###### data rotate ######
            data_rotate(saved_dir, data, img, 20, "_rotate_", saving_enable=True)

            ###### data flip and save #####
            data_flip(saved_dir, data, img, rate, 1, False)  # verical random flip
            data_flip(saved_dir, data, img, rate, 0, False)  # horizen random flip
            data_flip(saved_dir, data, img, rate, -1, False)  # both random flip

            ####### Image Scale #########
            if if_scale == True:
                print("Start Scale!")
                x = shape[0]
                y = shape[1]
                f_x = x + (x * (rate / 100))
                f_y = y + (y * (rate / 100))
                cv.resize(img, None, fx=f_x, fy=f_y, interpolation=cv.INTER_CUBIC)

                img = img[0:y, 0:x]

                save(saved_dir, data, img, rate, type)
            ############################

        plt.imshow(img)
        plt.show()

        return "success"

    except Exception as e:
        print(e)
        return "Failed"


def data_rotate(saved_dir, data, img, rate, type, saving_enable=False):
    xLength = img.shape[0]
    yLength = img.shape[1]

    try:
        rotation_matrix = cv.getRotationMatrix2D((xLength / 2, yLength / 2), rate, 1)
        img = cv.warpAffine(img, rotation_matrix, (xLength, yLength))
        # print(img.shape)
        if saving_enable == True:
            i = 0
            for i in range(10):
                save(saved_dir, data, img, rate, type)

        return "Success"
    except Exception as e:
        print(e)
        return "Failed"


def data_flip(saved_dir, data, img, rate, type, saving_enable=False):
    img = cv.flip(img, type)
    try:
        if type == 0:
            type = "_horizen_"
        elif type == 1:
            type = "_vertical_"
        elif type == -1:
            type = "_bothFlip_"

        if saving_enable == True:
            save(saved_dir, data, img, rate, type)

    except Exception as e:
        print(e)
        return "Failed"


def main_TransformImage(keyNames):
    try:
        for keyname in keyNames:
            # print(keyname)
            augmente(keyname, 10)  # scaling

        return "Augment Done!"
    except Exception as e:
        print(e)
        return "Augment Error!"


main_TransformImage(root_dir)
