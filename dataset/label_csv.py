'''
파일 기능: 이미지를 머신러닝을 위한 pickle 파일로 변환하기 위한 csv 파일 생성
작성자 : 유다영
작성일 : 2022.05.31
기타:
column이 2개로 되어 있던데, 원래 img이름과 class로 나눠야 하나?
신경망에서 class가 의미하는게 뭐지?

출처속 사진 보면 drop-0.png로 이름지어서 (-)앞에 있는 문자가(drop) class로 추출되는 것 같음

출처: https://www.notion.so/CNN-54af42a0bfa441ee90dc644416b78df6
'''

import os
import natsort
import csv
import re

# train.csv/ valid.csv/ test.csv 파일을 만들자.
file_path = 'handwriting_number/test/'  # 각 파일에 맞춰 여기 경로명 수정
file_lists = os.listdir(file_path)
file_lists = natsort.natsorted(file_lists)

f = open('test.csv', 'w', encoding='utf-8')  # 각 파일에 맞춰 여기 csv 파일명 변경
wr = csv.writer(f)
wr.writerow(["Img_name", "Class"])
for file_name in file_lists:
    print(file_name)
    wr.writerow([file_name, re.sub('-\d*[.]\w{3}', '', file_name)])
f.close()
