'''
파일 기능: opencv를 이용한 사진 내 숫자 부분 사각형으로 자르기
작성자 : 유다영
작성일 : 2022.06.03
기타: 12개의 숫자를 자르는게 안돼
왜일까?
이미지 어그멘테이션이라면 몸서리가 나는걸요?
저녁먹고 오겠음..

출처: https://pinkwink.kr/1129'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./dataset/test_opencv.PNG")

plt.figure(figsize=(15, 12))
plt.imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(15, 12))
plt.imshow(img_gray)

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
plt.figure(figsize=(15, 12))
plt.imshow(img_blur)

ret, img_th = cv2.threshold(img_blur, 100, 230, cv2.THRESH_BINARY_INV)

contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(each) for each in contours]
print(rects)

tmp = [w * h for (x, y, w, h) in rects]
tmp.sort()
print(tmp)

rects = [(x, y, w, h) for (x, y, w, h) in rects if ((w * h > 150) and (w * h < 5000))]
print(rects)

img_result = []
img_for_class = img.copy()

margin_pixel = 60

for rect in rects:
    # [y:y+h, x:x+w]
    img_result.append(
        img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel,
        rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel])

    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]),
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5)

plt.figure(figsize=(15, 12))
plt.imshow(img)

plt.figure(figsize=(4,4))
plt.imshow(img_result[0])

plt.figure(figsize=(4,4))
plt.imshow(cv2.resize(img_result[0], (28,28)))


