'''
파일 기능: 폴더 내 여러 폴더들의 파일들을 하나의 디렉토리 안에 옮기기
작성자 : 유다영
작성일 : 2022.06.07
출처: https://gagadi.tistory.com/9
'''

import os
import shutil
import time


def read_all_file(path):
    output = os.listdir(path)
    file_list = []

    for i in output:
        if os.path.isdir(path + "/" + i):
            file_list.extend(read_all_file(path + "/" + i))
        elif os.path.isfile(path + "/" + i):
            file_list.append(path + "/" + i)

    return file_list


def copy_all_file(file_list, new_path):
    for src_path in file_list:
        file = src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path + "/" + file)
        print("파일 {} 작업 완료".format(file))  # 작업한 파일명 출력


start_time = time.time()  # 작업 시작 시간

src_path = "C:\\Users\\DS\\OneDrive - 덕성여자대학교\\딥러닝기초\\Final_Project\\dataset\\handwriting_number\\augimg_all_file_sort"  # 기존 폴더 경로
src_path = src_path.replace('\\','/')

new_path = "C:\\Users\\DS\\OneDrive - 덕성여자대학교\\딥러닝기초\\Final_Project\\dataset\\handwriting_number\\augimg_all_file_sort\\all"  # 옮길 폴더 경로
new_path = new_path.replace('\\','/')

file_list = read_all_file(src_path)
copy_all_file(file_list, new_path)

print("=" * 40)
print("러닝 타임 : {}".format(time.time() - start_time))  # 총 소요시간 계산