'''
파일 기능: 폴더 내 파일 이름 변경
작성자 : 유다영
작성일 : 2022.05.31
기타:
코드 보완 필요, 기능 작동 X
'''
import os

def change_name(path, cName):
    i = 0
    for filename in os.listdir(path):
            print(path + filename, '=>', path + str(cName) + str(i) + '.png')
            os.rename(path + filename, path + str(cName) + str(i) + '.png')
            i += 1

change_name('dataset/handwriting_number/train/1', '1-')


