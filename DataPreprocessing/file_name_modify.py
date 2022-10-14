'''
파일 기능: 폴더 내 파일 이름 변경
작성자 : 유다영
작성일 : 2022.05.31
출처: https://hogni.tistory.com/35
'''
import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.

for i in range(0, 21):
    file_path = f'F:\\2022-1-Lecture\\딥러닝기초\\Final_Project\\dataset\\handwriting_number\\train\\{i}' # 폴더명 변경
    file_names = os.listdir(file_path)
    print(file_names)

    k = 0
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(k) + f'-{i-1}.png' # 순서에 맞게 숫자 변경 (폴더명-1)
        print(dst)
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        k += 1
