# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import sys

class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.image = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class mainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.points = []
        self.imageWidget = OwnImageWidget(self)
        self.posLabel = QLabel("Label")

        VBL = QVBoxLayout(self)
        VBL.addWidget(self.imageWidget)
        VBL.addWidget(self.posLabel)

        self.setMouseTracking(True)

        self.show()

    def mouseMoveEvent(self, event):
        self.points = event.pos()
        self.posLabel.setText(f"[{self.points.x()}, {self.points.y()}]")
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainWindow()
    sys.exit(app.exec_())
