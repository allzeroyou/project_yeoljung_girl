'''
파일 기능: opencv를 이용한 사진 내 숫자 부분 경계선 표시
작성자 : 유다영
작성일 : 2022.06.03
기타:
숫자 부분 경계선만 표시 가능.
하나씩 자르려면..?
opencv의 rectangle 사용하면 되려나
출처: https://www.zinnunkebi.com/python-opencv-drawcontours/
참고할 만 한 ref:
https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
https://copycoding.tistory.com/146
'''
import cv2
from matplotlib import pyplot as plt


def pause():
    # pause
    keycode = cv2.waitKey(0)
    # ESC key to close imshow
    if keycode == 27:
        cv2.destroyAllWindows()


img_bgr = cv2.imread('./dataset/test_opencv.PNG')
cv2.imshow("img_bgr", img_bgr)
pause()  # esc를 눌러 진행

img_bitwise_not_bgr = cv2.bitwise_not(img_bgr)
cv2.imshow("img_bitwise_not_bgr", img_bitwise_not_bgr)
pause()  # esc를 눌러 진행

img_bitwise_not_bgr2gray = cv2.cvtColor(img_bitwise_not_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_bitwise_not_bgr2gray", img_bitwise_not_bgr2gray)
pause()  # esc를 눌러 진행

ret, img_binary = cv2.threshold(img_bitwise_not_bgr2gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("img_binary", img_binary)
pause()  # esc를 눌러 진행

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contour = cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
cv2.imshow("img_contour", img_contour)
pause()  # esc를 눌러 진행

# cv2.imwrite('opencv_img_contour.png', img_contour)
