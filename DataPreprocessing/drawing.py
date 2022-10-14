'''
파일 기능: 그림판에서 자동으로 숫자를 적어줌
작성자 : 유다영
작성일 : 2022.05.26
기타: 인터넷에서 긁어옴, 코드를 우리 플젝에 맞게 수정 및 코드 이해 필요함, 아직 기능 작동 X
'''

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import tensorflow as tf


# 2. 그림판 렌더링 클래스
class MyApp(QMainWindow):

    # 3. 그림판 초기 설정 함수
    def __init__(self):
        super().__init__()
        self.statusbar = None
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 30
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.loaded_model = None
        self.initUI()

    def initUI(self):
        # 상단 메뉴바 렌더링
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        # 학습된 분류 모델 로드
        load_model_action = QAction('Load model', self)
        load_model_action.setShortcut('Ctrl+L')
        load_model_action.triggered.connect(self.load_model)

        # 드로잉 사진 저장
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)

        # 드로잉 내용 초기화
        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()

        # 4. 그림판 렌더링
        self.setWindowTitle('MNIST Classifier')
        self.setGeometry(300, 300, 400, 400)
        self.show()

    # 5. 드로잉 함수
    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    # 6. 마우스 이벤트 함수
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(
                QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    # 7. 드로잉 내용 숫자 인식 및 판별 함수
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

            arr = np.zeros((28, 28))
            for i in range(28):
                for j in range(28):
                    arr[j, i] = 1 - \
                                self.image.scaled(28, 28).pixelColor(
                                    i, j).getRgb()[0] / 255.0
            arr = arr.reshape(-1, 28, 28)

            if self.loaded_model:
                pred = self.loaded_model.predict(arr)[0]
                pred_num = str(np.argmax(pred))
                self.statusbar.showMessage('숫자 ' + pred_num + '입니다.')

    # 학습된 분류 모델 로드 실행 함수
    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Model', '')

        if fname:
            self.loaded_model = tf.keras.models.load_model(fname)
            self.statusbar.showMessage('Model loaded.')

    # 드로잉 사진 저장 실행 함수
    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if fpath:
            self.image.scaled(28, 28).save(fpath)

    # 드로잉 내용 초기화 실행 함수
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        self.statusbar.clearMessage()


# 1. 메인 함수 시작
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()  # 그림판 렌더링 호출
    sys.exit(app.exec_())
