# -*- coding: utf-8 -*-

import ctypes as ct
import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
import logging
from glob import glob
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir, Qt, QSettings, QTimer
from PyQt5 import QtGui
from PyQt5 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg\
 import FigureCanvasQTAgg as FigureCanvas


os.chdir(os.path.dirname(__file__))
os.chdir(r'../Andor_dll/Andor_dll')
# os.chdir(r'Andor_dll\Andor_dll')

class nsPrms(ct.Structure):
    _fields_ = [("norm", ct.c_bool),
                ("normMax", ct.c_double),
                ("normMin", ct.c_double),
                ("sigma", ct.c_double)]


class blurPrms(ct.Structure):
    _fields_ = [("exe", ct.c_bool),
                ("sizeX", ct.c_int),
                ("sizeY", ct.c_int),
                ("sigma", ct.c_double)]


class thresPrms(ct.Structure):
    _fields_ = [("exe", ct.c_bool),
                ("thres", ct.c_double),
                ("maxVal", ct.c_double),
                ("type", ct.c_int)]


class contPrms(ct.Structure):
    _fields_ = [("exe", ct.c_bool),
                ("num", ct.c_int)]


class processPrms(ct.Structure):
    _fields_ = [("ns", nsPrms),
                ("blur", blurPrms),
                ("thres", thresPrms),
                ("cont", contPrms)]

dll = ct.windll.LoadLibrary(r'..\x64\Release\Andor_dll.dll')
dll.InitialiseLibrary()
dll.InitialiseUtilityLibrary()
dll.GetInt.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_longlong)]
dll.SetInt.argtypes = [ct.c_int, ct.c_wchar_p, ct.c_longlong]
dll.SetEnumString.argtypes = [ct.c_int, ct.c_wchar_p, ct.c_wchar_p]
dll.GetEnumIndex.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_int)]
dll.SetFloat.argtypes = [ct.c_int, ct.c_wchar_p, ct.c_double]
dll.GetFloat.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_double)]
dll.GetFloatMax.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_double)]
dll.GetFloatMin.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_double)]
dll.SetBool.argtypes = [ct.c_int, ct.c_wchar_p, ct.c_bool]
dll.GetBool.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_bool)]
dll.Command.argtypes = [ct.c_int, ct.c_wchar_p]
dll.WaitBuffer.argtypes = [ct.c_int, ct.POINTER(ct.POINTER(ct.c_ubyte)), ct.POINTER(ct.c_int), ct.c_uint]
dll.QueueBuffer.argtypes = [ct.c_int, ct.POINTER(ct.c_ubyte), ct.c_int]
dll.Flush.argtypes = [ct.c_int]
dll.convertBuffer.argtypes = [ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_ushort), ct.c_longlong, ct.c_longlong, ct.c_longlong]
dll.processImageShow.argtypes = (ct.c_longlong, ct.c_longlong, ct.POINTER(ct.c_ushort),
                                 processPrms, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
dll.processImage.argtypes = (ct.POINTER(ct.c_float), ct.c_longlong, ct.c_longlong, ct.POINTER(ct.c_ushort), processPrms)

# Class definition for multithreading
class ImageAcquirer(QtCore.QThread):

    posSignal = QtCore.pyqtSignal(float, float)
    imgSignal = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stopped = False
        self.mutex = QtCore.QMutex()

    def setup(self, Handle, dir=None, num=100, count_p=None, center=None,
              DIOthres=0, DIOhandle=None, fixed=0, mode=0):
        self.Handle = Handle
        self.dir = dir
        self.num = num
        self.count_p = count_p
        self.center = (ct.c_float*2)(0.0, 0.0)
        if not(center is None):
            self.center = (ct.c_float*2)(center[0], center[1])
        self.DIOthres = ct.c_float(DIOthres)
        self.DIOhandle = DIOhandle
        self.fixed = fixed
        self.mode = mode
        self.stopped = False

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        if self.stopped:
            return
        if self.fixed == 0:
            if self.mode==0:
                dll.startFixedAcquisition(self.Handle, self.dir, self.num, self.count_p)
            elif self.mode==1:
                self.prepareAcquisition()
            else:
                self.feedbackedAcquisition()
        else:
            self.continuousAcquisition()
        self.stop()
        self.finished.emit()

    def continuousAcquisition(self):
        ImageSizeBytes = ct.c_longlong()
        dll.GetInt(self.Handle, "ImageSizeBytes", ct.byref(ImageSizeBytes))
        NumberOfBuffers = 10
        UserBuffers = []
        for i in range(NumberOfBuffers):
            UserBuffers.append((ct.c_ubyte*ImageSizeBytes.value)())
            dll.QueueBuffer(self.Handle, UserBuffers[i], ct.c_int(ImageSizeBytes.value))

        ImageHeight = ct.c_longlong()
        ImageWidth = ct.c_longlong()
        ImageStride = ct.c_longlong()

        dll.GetInt(self.Handle, "AOI Height", ct.byref(ImageHeight))
        dll.GetInt(self.Handle, "AOI Width", ct.byref(ImageWidth))
        dll.GetInt(self.Handle, "AOI Stride", ct.byref(ImageStride))

        dll.SetEnumString(self.Handle, "CycleMode", "Continuous")
        dll.SetFloat(self.Handle, "FrameRate", 30)
        dll.Command(self.Handle, "Acquisition Start")

        Buffer = ct.pointer(ct.c_ubyte())
        BufferSize = ct.c_int()
        outputBuffer = (ct.c_ushort * (ImageWidth.value * ImageHeight.value))()
        outBuffer = (ct.c_ubyte*(ImageWidth.value*ImageHeight.value))()
        maxVal = ct.c_double()
        minVal = ct.c_double()
        while not self.stopped:
            if dll.WaitBuffer(self.Handle, ct.byref(Buffer), ct.byref(BufferSize), 10000) == 0:
                ret = dll.convertBuffer(Buffer, outputBuffer, ImageWidth, ImageHeight, ImageStride)
                dll.processImageShow(ImageHeight, ImageWidth, outputBuffer, self.prms,
                                     outBuffer, ct.byref(maxVal), ct.byref(minVal))
                self.max = maxVal.value
                self.min = minVal.value
                self.img = np.array(outBuffer).reshape(ImageHeight.value, ImageWidth.value)
                self.width = ImageWidth.value
                self.height = ImageHeight.value
                self.imgSignal.emit(self)
                dll.QueueBuffer(self.Handle, Buffer, BufferSize)
            else:
                logging.error('failed')
                break
        dll.Command(self.Handle, "Acquisition Stop")
        del UserBuffers
        del outputBuffer
        del outBuffer
        dll.Flush(self.Handle)

    def prepareAcquisition(self):

        point = (ct.c_float*2)()
        dll.multithread(self.Handle, self.dir, self.num, self.count_p, self.prms,
        point, self.center, self.DIOthres, self.DIOhandle, ct.c_int(self.mode))
        self.posSignal.emit(point[0], point[1])

    def feedbackedAcquisition(self):
        point = (ct.c_float*2)()
        dll.multithread(self.Handle, self.dir, self.num, self.count_p, self.prms,
        point, self.center, self.DIOthres, self.DIOhandle, ct.c_int(self.mode))


class ImagePlayer(QtCore.QThread):
    countStepSignal = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.stopped = False
        self.mutex = QtCore.QMutex()

    def setup(self):
        self.stopped = False

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        if self.stopped:
            return
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.countStep)
        self.timer.start(50)
        self.exec_()
        self.stop()
        self.finished.emit()

    def countStep(self):
        if self.stopped:
            self.exit()
        else:
            self.countStepSignal.emit()


class ImageProcesser(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stopped = False
        self.mutex = QtCore.QMutex()

    def setup(self, datfiles, width, height, stride, prms, dir, pBar):
        self.datfiles = datfiles
        self.width = width
        self.height = height
        self.stride = stride
        self.prms = prms
        self.dir = dir
        self.pBar = pBar
        self.stopped = False

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        if self.stopped:
            return
        ImgArry = (ct.c_ushort * (self.width * self.height))()
        point = (ct.c_float*2)()
        pointlst = []
        num = len(self.datfiles)
        self.pBar.setMaximum(num-1)
        for datfile, i in zip(self.datfiles, range(num)):
            rawdata = np.fromfile(datfile, dtype=np.uint8)
            buffer = rawdata.ctypes.data_as(ct.POINTER(ct.c_ubyte))
            ret = dll.convertBuffer(buffer, ImgArry, self.width, self.height, self.stride)
            dll.processImage(point, self.height, self.width, ImgArry, self.prms)
            pointlst.append([point[0], point[1]])
            self.pBar.setValue(i)
        DF = pd.DataFrame(np.array(pointlst))
        DF.columns = ['x', 'y']
        DF.to_csv(self.dir + r"\COG.csv")
        logging.info("analysis finished")


# class definition for UI
class LHLayout(QHBoxLayout):
    def __init__(self, label, object, parent=None):
        super(QHBoxLayout, self).__init__(parent)
        self.addWidget(QLabel(label), alignment=Qt.AlignRight)
        if isinstance(object, list):
            for Widget in object:
                self.addWidget(Widget)
        elif isinstance(object, QLayout):
            self.addLayout(object)
        else:
            self.addWidget(object)


class OwnImageWidget(QWidget):
    posSignal = QtCore.pyqtSignal(QtCore.QPoint)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.setMouseTracking(True)
        self.setMinimumSize(700, 700)

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

    def mouseMoveEvent(self, event):
        self.posSignal.emit(event.pos())

class PosLabeledImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.imageWidget = OwnImageWidget(self)
        self.posLabel = QLabel("Position:")

        self.imageWidget.posSignal.connect(self.writeLabel)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.imageWidget)
        vbox.addWidget(self.posLabel)
        vbox.addStretch(1)


    def writeLabel(self, pos):
        self.posLabel.setText(f"Position:[{pos.x()}, {pos.y()}]")
        self.update()

    def setImage(self, image):
        self.imageWidget.image = image
        sz = image.size()
        self.imageWidget.setMinimumSize(sz)
        self.update()


class SLMWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = OwnImageWidget(self)
        self.img = None

    def update_SLM(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.canvas.setImage(image)


class contWidget(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.initUI()

    def initUI(self):
        self.imgMaxBox = QLineEdit(self)
        self.imgMaxBox.setReadOnly(True)
        self.imgMaxBox.setSizePolicy((QSizePolicy(5, 0)))
        self.imgMinBox = QLineEdit(self)
        self.imgMinBox.setReadOnly(True)
        self.imgMinBox.setSizePolicy((QSizePolicy(5, 0)))
        self.markerPositionBoxX = QSpinBox(self)
        self.markerPositionBoxX.setMaximum(400)
        self.markerPositionBoxX.setMinimum(-400)
        self.markerPositionBoxY = QSpinBox(self)
        self.markerPositionBoxY.setMaximum(400)
        self.markerPositionBoxY.setMinimum(-400)
        self.markerFactorBox = QSpinBox(self)
        self.markerFactorBox.setRange(0, 9999)
        self.splitButton = QPushButton("Split", self)
        self.splitButton.setCheckable(True)

        self.initLayout()

    def initLayout(self):

        hbox00 = QVBoxLayout()
        hbox00.addWidget(QLabel("Raw Image"))
        hbox00.addLayout(LHLayout('Min: ', self.imgMinBox))
        hbox00.addLayout(LHLayout('Max: ', self.imgMaxBox))

        hbox01 = QVBoxLayout()
        hbox01.addWidget(QLabel("Marker Position"))
        hbox01.addLayout(LHLayout('x: ', self.markerPositionBoxX))
        hbox01.addLayout(LHLayout('y: ', self.markerPositionBoxY))
        hbox01.addLayout(LHLayout('factor: ', [self.markerFactorBox, self.splitButton]))

        hbox0 = QHBoxLayout(self)
        hbox0.addLayout(hbox00)
        hbox0.addLayout(hbox01)
        hbox0.setStretch(0, 1)
        hbox0.setStretch(1, 1)


class fixedWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.initUI()

    def initUI(self):
        self.numImgBox = QLineEdit('100', self)
        self.frameRateBox = QLineEdit('30', self)
        self.countBox = QLineEdit(self)
        self.countBox.setReadOnly(True)
        self.dirBox = QLineEdit(self)
        self.dirBox.setReadOnly(True)
        self.dirButton = QPushButton('...', self)
        self.dirButton.clicked.connect(self.selectDirectory)
        self.aqTypeBox = QComboBox(self)
        aqTypeList = ['No feed-back', 'Preparation', 'Do feed-back']
        self.aqTypeBox.addItems(aqTypeList)
        self.cntrPosLabel = QLabel("Centre of position:", self)
        self.thresBox = QDoubleSpinBox(self)
        self.periodButton = QPushButton('Periodic', self)
        self.periodButton.setCheckable(True)
        self.repeatBox = QSpinBox(self)
        self.repeatBox.setMinimum(1)
        self.currentRepeatBox = QLineEdit(self)
        self.currentRepeatBox.setReadOnly(True)
        self.currentRepeatBox.setSizePolicy(QSizePolicy(5, 0))

        self.initLayout()

    def initLayout(self):
        hbox00 = QHBoxLayout()
        hbox00.addLayout(LHLayout("Frames", self.numImgBox))
        hbox00.addLayout(LHLayout("Frame Rate", self.frameRateBox))
        hbox00.addLayout(LHLayout("Directory: ", [self.dirBox, self.dirButton]))
        hbox00.addLayout(LHLayout("Current Frame", self.countBox))

        hbox012 = LHLayout("Threshold: ", self.thresBox)
        hbox01 = QHBoxLayout()
        hbox01.addWidget(self.aqTypeBox)
        hbox01.addWidget(self.cntrPosLabel)
        hbox01.addLayout(hbox012)

        hbox021 = LHLayout("Repeat: ", self.repeatBox)
        hbox02 = QHBoxLayout()
        hbox02.addWidget(self.periodButton)
        hbox02.addLayout(hbox021)
        hbox02.addWidget(self.currentRepeatBox)

        vbox0 = QVBoxLayout(self)
        vbox0.addLayout(hbox00)
        vbox0.addLayout(hbox01)
        vbox0.addLayout(hbox02)

    def selectDirectory(self):
        self.dirname = QFileDialog.getExistingDirectory(self, 'Select directory', self.dirname)
        self.dirname = QDir.toNativeSeparators(self.dirname)
        self.dirBox.setText(self.dirname)


class AcquisitionWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.contWidget = contWidget(self)
        self.fixedWidget = fixedWidget(self)

        self.Tab = QTabWidget()
        self.Tab.addTab(self.contWidget, 'Continuous')
        self.Tab.addTab(self.fixedWidget, 'Fixed')

        self.initUI()

    def initUI(self):
        self.handleBox = QLineEdit(self)
        self.handleBox.setReadOnly(True)
        self.handleBox.setSizePolicy(QSizePolicy(5, 0))
        self.exposeTBox = QDoubleSpinBox(self)
        self.exposeTBox.setDecimals(5)
        self.AOISizeBox = QComboBox(self)
        AOISizeList = ['2048x2048','1024x1024','512x512','256x256','128x128','free']
        self.AOISizeBox.addItems(AOISizeList)
        self.AOILeftBox = QSpinBox(self)
        self.AOITopBox = QSpinBox(self)
        self.AOIWidthBox = QSpinBox(self)
        self.AOIHeightBox = QSpinBox(self)
        AOIBoxList = [self.AOITopBox, self.AOILeftBox, self.AOIWidthBox, self.AOIHeightBox]
        for obj in AOIBoxList:
            obj.setMaximum(2048)
        self.CenterXBox = QSpinBox(self)
        self.CenterYBox = QSpinBox(self)
        self.CenterXBox.setMinimum(-1024)
        self.CenterYBox.setMinimum(-1024)
        self.CenterXBox.setMaximum(1024)
        self.CenterYBox.setMaximum(1024)
        self.AOIBinBox = QComboBox(self)
        AOIBinList = ["1x1", "2x2", "3x3", "4x4", "8x8"]
        self.AOIBinBox.addItems(AOIBinList)
        self.globalClearButton = QPushButton('Global clear', self)
        self.globalClearButton.setCheckable(True)

        self.initButton = QPushButton("Initialize", self)
        self.applyButton = QPushButton('APPLY', self)
        self.runButton = QPushButton('RUN', self)
        self.finButton = QPushButton("Finalize", self)
        self.applyButton.setEnabled(False)
        self.runButton.setCheckable(True)
        self.runButton.setEnabled(False)
        self.finButton.setEnabled(False)

        self.initLayout()

    def initLayout(self):
        hbox00 = QHBoxLayout()
        hbox00.addLayout(LHLayout("Handle: ", self.handleBox))
        hbox00.addLayout(LHLayout('Exposure Time (s): ', self.exposeTBox))
        hbox00.addLayout(LHLayout("Binnng: ", self.AOIBinBox))
        hbox00.addWidget(self.globalClearButton)

        hbox01 = QHBoxLayout()
        hbox01.addLayout(LHLayout("AOI Size: ", self.AOISizeBox))
        hbox01.addLayout(LHLayout("Top: ", self.AOITopBox))
        hbox01.addLayout(LHLayout("Left: ", self.AOILeftBox))
        hbox01.addLayout(LHLayout("Width: ", self.AOIWidthBox))
        hbox01.addLayout(LHLayout("Height: ", self.AOIHeightBox))
        hbox01.addLayout(LHLayout("X: ", self.CenterXBox))
        hbox01.addLayout(LHLayout("Y: ", self.CenterYBox))

        hbox03 = QHBoxLayout()
        hbox03.addWidget(self.initButton)
        hbox03.addWidget(self.applyButton)
        hbox03.addWidget(self.runButton)
        hbox03.addWidget(self.finButton)

        vbox0 = QVBoxLayout(self)
        vbox0.addLayout(hbox00)
        vbox0.addLayout(hbox01)
        vbox0.addWidget(self.Tab)
        vbox0.addLayout(hbox03)


class imageLoader(QWidget):

    imgSignal = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()
        self.initVal()

    def initUI(self):

        self.dirBox = QLineEdit(self)
        self.dirBox.setReadOnly(True)
        self.dirBox.setSizePolicy(QSizePolicy(7, 0))
        self.dirButton = QPushButton('...', self)
        self.dirButton.clicked.connect(self.selectDirectory)
        self.fileNumBox = QLineEdit(self)
        self.fileNumBox.setReadOnly(True)
        self.fileNumBox.setSizePolicy(QSizePolicy(5, 0))
        self.currentNumBox = QSpinBox(self)
        self.currentNumBox.valueChanged.connect(self.update_img)
        self.currentNumBox.setWrapping(True)

        self.imgMaxBox = QLineEdit(self)
        self.imgMaxBox.setReadOnly(True)
        self.imgMaxBox.setSizePolicy(QSizePolicy(5, 0))
        self.imgMinBox = QLineEdit(self)
        self.imgMinBox.setReadOnly(True)
        self.imgMinBox.setSizePolicy(QSizePolicy(5, 0))

        self.anlzButton = QPushButton('Analysis', self)
        self.anlzStartBox = QSpinBox(self)
        self.anlzEndBox = QSpinBox(self)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)

        self.initLayout()

    def initLayout(self):
        hbox01 = QHBoxLayout()
        hbox01.addLayout(LHLayout("Frames: ", self.fileNumBox))
        hbox01.addLayout(LHLayout("Current Frame: ", self.currentNumBox))

        hbox02 = QHBoxLayout()
        hbox02.addLayout(LHLayout("Min: ", self.imgMinBox))
        hbox02.addLayout(LHLayout("Max: ", self.imgMaxBox))

        hbox03 = QHBoxLayout()
        hbox03.addWidget(self.anlzButton)
        hbox03.addLayout(LHLayout("Start: ", self.anlzStartBox))
        hbox03.addLayout(LHLayout("End: ", self.anlzEndBox))

        vbox0 = QVBoxLayout(self)
        vbox0.addLayout(LHLayout("Directory: ", [self.dirBox, self.dirButton]))
        vbox0.addLayout(hbox01)
        vbox0.addLayout(hbox02)
        vbox0.addLayout(hbox03)
        vbox0.addWidget(self.progressBar)

    def initVal(self):
        self.dirname = None
        self.img = None

    def selectDirectory(self):
        dirname = QFileDialog.getExistingDirectory(self, 'Select directory', self.dirname)
        dirname = QDir.toNativeSeparators(dirname)
        self.dirBox.setText(dirname)
        metafile = dirname + r'\metaSpool.txt'
        if os.path.isfile(metafile):
            self.dirname = dirname
            f = open(metafile, mode='r')
            metadata = f.readlines()
            f.close()
            self.datfiles = glob(dirname + r"\*.dat")
            self.filenum = len(self.datfiles)
            self.fileNumBox.setText(str(self.filenum))
            self.anlzStartBox.setMaximum(self.filenum-1)
            self.anlzEndBox.setMaximum(self.filenum-1)
            self.anlzEndBox.setValue(self.filenum-1)
            self.imgSize = int(metadata[0])
            self.encoding = metadata[1]
            self.stride = int(metadata[2])
            self.height = int(metadata[3])
            self.width = int(metadata[4])
            self.currentNumBox.setMaximum(self.filenum - 1)
            self.update_img()
        else:
            logging.error('No metadata was found.\n')

    def update_frame(self):
        img = self.img
        img_height, img_width = img.shape
        scale_w = float(800) / float(img_width)
        scale_h = float(600) / float(img_height)
        scale = min([scale_w, scale_h])

        if scale == 0:
            scale = 1

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = cv2.convertScaleAbs(img, alpha=1/16)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        window.central.ImgWidget.setImage(image)


    def update_img(self):
        num = int(self.currentNumBox.text())
        rawdata = np.fromfile(self.datfiles[num], dtype=np.uint8)
        buffer = rawdata.ctypes.data_as(ct.POINTER(ct.c_ubyte))
        outputBuffer = (ct.c_ushort * (self.width * self.height))()
        outBuffer = (ct.c_ubyte*(self.width*self.height))()
        max = ct.c_double()
        min = ct.c_double()
        ret = dll.convertBuffer(buffer, outputBuffer, self.width, self.height, self.stride)
        dll.processImageShow(self.height, self.width, outputBuffer, self.prms, outBuffer,
                             ct.byref(max), ct.byref(min))
        self.min = min.value
        self.max = max.value
        self.img = np.array(outBuffer).reshape(self.height, self.width)
        self.imgMinBox.setText(str(self.min))
        self.imgMaxBox.setText(str(self.max))
        self.imgSignal.emit(self)


class processWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()
        self.initVal()

    def initUI(self):
        self.normButton = QRadioButton('Normalize', self)
        self.normButton.setChecked(True)
        self.normMax = QSpinBox(self)
        self.normMax.setRange(0, 4095)
        self.normMax.setValue(4095)
        self.normMin = QSpinBox(self)
        self.normMin.setRange(0, 4095)
        self.standButton = QRadioButton('Standardize', self)
        self.sigma = QSpinBox(self)
        self.sigma.setRange(0, 255)
        self.sigma.setValue(64)
        self.blurButton = QRadioButton('Blur', self)
        self.blurButton.setAutoExclusive(False)
        self.blurSize = QSpinBox(self)
        self.blurSigma = QSpinBox(self)

        self.thresButton = QRadioButton('Threshold', self)
        self.thresButton.setAutoExclusive(False)
        self.thresVal = QSpinBox(self)
        self.thresVal.setRange(0, 255)
        self.thresType = QComboBox(self)
        typelist = ['cv2.THRESH_BINARY', 'cv2.THRESH_BINARY_INV',
        'cv2.THRESH_TRUNC', 'cv2.THRESH_TOZERO',
        'cv2.THRESH_TOZERO_INV']
        self.thresType.addItems(typelist)
        self.OtsuButton = QRadioButton('Otsu', self)

        self.contButton = QRadioButton('Find Contours', self)
        self.contButton.setAutoExclusive(False)
        self.contNum = QSpinBox(self)
        self.contNum.setRange(1,4)

        self.applyButton = QPushButton('APPLY')

        self.initLayout()

    def initLayout(self):
        hbox000 = QHBoxLayout()
        hbox000.addWidget(self.normButton)
        hbox000.addLayout(LHLayout("Min: ", self.normMin))
        hbox000.addLayout(LHLayout("Max: ", self.normMax))

        hbox001 = QHBoxLayout()
        hbox001.addWidget(self.standButton)
        hbox001.addLayout(LHLayout("Sigma: ", self.sigma))

        hbox00 = QHBoxLayout()
        hbox00.addLayout(hbox000)
        hbox00.addLayout(hbox001)

        hbox01 = QHBoxLayout()
        hbox01.addWidget(self.blurButton)
        hbox01.addLayout(LHLayout("Kernel size : ", self.blurSize))
        hbox01.addLayout(LHLayout("Sigma: ", self.blurSigma))

        hbox02 = QHBoxLayout()
        hbox02.addWidget(self.thresButton)
        hbox02.addLayout(LHLayout("Threshold: ", self.thresVal))
        hbox02.addLayout(LHLayout("Type: ", self.thresType))
        hbox02.addWidget(self.OtsuButton)

        hbox03 = QHBoxLayout()
        hbox03.addWidget(self.contButton)
        hbox03.addLayout(LHLayout("Number to Find: ", self.contNum))

        vbox0 = QVBoxLayout(self)
        vbox0.addLayout(hbox00)
        vbox0.addLayout(hbox01)
        vbox0.addLayout(hbox02)
        vbox0.addLayout(hbox03)
        vbox0.addWidget(self.applyButton)

        # self.setStyleSheet("background-color:white;")
        self.setTitle("Process Settings")

    def initVal(self):
        self.prmStruct = processPrms(nsPrms(0,0,0,0),
        blurPrms(0,0,0,0),
        thresPrms(0,0,0,0),
        contPrms(0,0))

    def set_prm(self):
        self.prmStruct.ns = nsPrms(self.normButton.isChecked(), self.normMax.value(), self.normMin.value(), self.sigma.value())
        self.prmStruct.blur = blurPrms(self.blurButton.isChecked(), self.blurSize.value(), self.blurSize.value(), self.blurSigma.value())
        self.prmStruct.thres = thresPrms(self.thresButton.isChecked(), self.thresVal.value(), 255, self.thresType.currentIndex()+int(self.OtsuButton.isChecked())*8)
        self.prmStruct.cont = contPrms(self.contButton.isChecked(), self.contNum.value())


class SLM_Controller(QGroupBox):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.w = SLMWindow(self)
        self.initUI()
        self.initVal()

    def initUI(self):
        self.SLMButton = QPushButton('SLM ON', self)
        self.SLMButton.setCheckable(True)
        self.SLMButton.toggled.connect(self.switch_SLM)

        self.modulateButton = QPushButton('Modulation', self)
        self.modulateButton.setCheckable(True)
        self.modulateButton.toggled.connect(self.modulate_SLM)

        self.wavelengthBox = QComboBox(self)
        self.wavelengthBox.addItems(("1050nm", "1060nm", "1064nm"))
        self.wavelengthBox.currentIndexChanged.connect(self.wavelengthChanged)

        self.pitchBox = QSpinBox(self)
        self.pitchBox.setMinimum(1)

        self.SLMDial = QDial(self)
        self.SLMDial.setWrapping(True)
        self.SLMDial.setMaximum(360)
        self.SLMDial.setMinimum(0)

        self.focusBox = QDoubleSpinBox(self)
        self.focusBox.setRange(-3.0, 3.0)
        self.focusBox.setSingleStep(0.01)
        self.focusBox.setDecimals(3)
        self.focusBox.valueChanged.connect(self.focusChanged)

        self.focusXBox = QSpinBox(self)
        self.focusXBox.setRange(-396, 396)
        self.focusYBox = QSpinBox(self)
        self.focusYBox.setRange(-300, 300)
        self.focusXBox.valueChanged.connect(self.focusChanged)
        self.focusYBox.valueChanged.connect(self.focusChanged)

        self.initLayout()

    def initLayout(self):
        hbox010 = QHBoxLayout()
        hbox010.addWidget(self.SLMButton)
        hbox010.addWidget(self.modulateButton)

        hbox011 = LHLayout("Wavelength: ", self.wavelengthBox)
        hbox012 = LHLayout("Pitch: ", self.pitchBox)
        hbox013 = LHLayout("Focus: ", self.focusBox)

        hbox014 = QHBoxLayout()
        hbox014.addWidget(QLabel('FocusPoint:'), alignment=Qt.AlignRight)
        hbox014.addWidget(self.focusXBox)
        hbox014.addWidget(self.focusYBox)

        vbox01 = QVBoxLayout()
        vbox01.addLayout(hbox010)
        vbox01.addLayout(hbox011)
        vbox01.addLayout(hbox012)
        vbox01.addLayout(hbox013)
        vbox01.addLayout(hbox014)

        hbox0 = QHBoxLayout(self)
        hbox0.addWidget(self.SLMDial)
        hbox0.addLayout(vbox01)

        self.setTitle("SLM Controller")

    def initVal(self):
        self.correctImg = [r'CorrectImages\CAL_LSH0701554_1050nm.bmp', \
        r'CorrectImages\CAL_LSH0701554_1060nm.bmp', r'CorrectImages\CAL_LSH0701554_1064nm.bmp']
        self.alphaList = [214, 216, 217]
        self.correction = cv2.imread(self.correctImg[0], 0)
        self.alpha = self.alphaList[0]
        self.base = self.correction
        self.theta = 0
        self.pitch = 23
        self.focus = 0
        self.focusX = 396
        self.focusY = 300

    def switch_SLM(self, checked):
        if checked:
            desktop = app.desktop()
            left = desktop.availableGeometry(1).left()
            top = desktop.availableGeometry(1).top()
            self.focus = self.focusBox.value()
            self.w.move(left, top)
            self.w.showFullScreen()
            self.make_base()
            self.init_img()
            self.w.update_SLM()
        else:
            self.w.reject()

    def wavelengthChanged(self, index):
        self.correction = cv2.imread(self.correctImg[index], 0)
        self.alpha = self.alphaList[index]

    def make_base(self):
        x = np.arange(792)
        y = np.arange(600)
        X, Y = np.meshgrid(x, y)
        focusCorrection = ((X - self.focusX)**2 + (Y - self.focusY)**2)*self.focus*1e-2
        self.base = self.correction + focusCorrection.astype(np.uint8)

    def init_img(self):
        img = self.base.astype(np.float) * self.alpha / 256
        self.w.img = img.astype(np.uint8)

    def update_img(self):
        x = np.arange(792)
        y = np.arange(600)
        X, Y = np.meshgrid(x, y)

        rotX = X*np.sin(self.theta) + Y*np.cos(self.theta)
        rotY = X*np.cos(self.theta) - Y*np.sin(self.theta)

        img = ((rotX // self.pitch + rotY // self.pitch) % 2) * 128
        img = (img + self.base).astype(np.uint8)
        img = img.astype(np.float) * self.alpha / 256
        self.w.img = img.astype(np.uint8)
        self.w.update_SLM()

    def modulate_SLM(self, checked):
        if checked:
            self.update_img()
        else:
            self.init_img()
            self.w.update_SLM()

    def focusChanged(self, val):
        self.focus = self.focusBox.value()
        self.focusX = self.focusXBox.value()+396
        self.focusY = self.focusYBox.value()+300
        self.make_base()
        if self.SLMButton.isChecked():
            if self.modulateButton.isChecked():
                self.update_img()
            else:
                self.init_img()
                self.w.update_SLM()


class DIOWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.DIOHighButton = QPushButton('High', self)
        self.DIOLowButton = QPushButton('Low', self)

        self.DIOHighButton.setCheckable(True)
        self.DIOLowButton.setCheckable(True)

        self.DIObuttons = QButtonGroup(self)
        self.DIObuttons.addButton(self.DIOHighButton)
        self.DIObuttons.addButton(self.DIOLowButton)
        self.DIObuttons.setExclusive(True)
        self.DIObuttons.setId(self.DIOLowButton, 6)
        self.DIObuttons.setId(self.DIOHighButton, 7)

        vbox = QVBoxLayout(self)
        # vbox.addWidget(QLabel('DIO Control'))
        vbox.addWidget(self.DIOHighButton)
        vbox.addWidget(self.DIOLowButton)

        self.DIOLowButton.setChecked(True)
        self.setTitle("DIO Controller")


class shutterWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.openButton = QPushButton('Open', self)
        self.closeButton = QPushButton('Close', self)
        self.openButton.setCheckable(True)
        self.closeButton.setCheckable(True)

        self.shutterButtons = QButtonGroup(self)
        self.shutterButtons.addButton(self.openButton)
        self.shutterButtons.addButton(self.closeButton)
        self.shutterButtons.setExclusive(True)
        self.shutterButtons.setId(self.openButton, 5)
        self.shutterButtons.setId(self.closeButton, 4)

        vbox = QVBoxLayout(self)
        # vbox.addWidget(QLabel('DIO Control'))
        vbox.addWidget(self.openButton)
        vbox.addWidget(self.closeButton)

        self.closeButton.setChecked(True)
        self.setTitle("Shutter Controller")

class Logger(logging.Handler):
    def __init__(self, parent):
     super().__init__()
     self.widget = QPlainTextEdit(parent)
     self.widget.setReadOnly(True)

    def emit(self, record):
     msg = self.format(record)
     self.widget.appendPlainText(msg)


class LogWidget(QGroupBox):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        vbox = QVBoxLayout(self)

        logTextBox = Logger(self)
# You can format what is logging.infoed to text box
        logTextBox.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
# You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        vbox.addWidget(logTextBox.widget)
        self.setLayout(vbox)
        self.setTitle("Log")


class centralWidget(QWidget):

    def __init__(self, parent=None):
        super(centralWidget, self).__init__(parent)

        self.imageAcquirer = ImageAcquirer()
        self.imagePlayer = ImagePlayer()
        self.imageProcesser = ImageProcesser()
        self.ImgWidget = PosLabeledImageWidget(self)
        self.SLM_Controller = SLM_Controller(self)
        self.imageLoader = imageLoader(self)
        self.processWidget = processWidget(self)
        self.DIOWidget = DIOWidget(self)
        self.shutterWidget = shutterWidget(self)
        self.logWidget = LogWidget(self)
        self.acquisitionWidget = AcquisitionWidget(self)

        self.modeTab = QTabWidget()
        self.modeTab.addTab(self.acquisitionWidget, 'Image Acquisition')
        self.modeTab.addTab(self.imageLoader, 'Image Load')

        self.initSignal()
        self.initUI()
        self.initVal()

    def initSignal(self):
        self.imageAcquirer.imgSignal.connect(self.update_frame)
        self.imageAcquirer.posSignal.connect(self.applyCenter)
        # self.imageAcquirer.finished.connect(self.acquisitionFinished)

        self.imagePlayer.countStepSignal.connect(self.imageLoader.currentNumBox.stepUp)
        self.imageLoader.anlzButton.clicked.connect(self.analyzeDatas)
        self.imageLoader.imgSignal.connect(self.update_frame)

        self.processWidget.applyButton.clicked.connect(self.applyProcessClicked)

        self.DIOWidget.DIObuttons.buttonClicked[int].connect(self.writeDIO)

        self.shutterWidget.shutterButtons.buttonClicked[int].connect(self.writeDIO)

        self.acquisitionWidget.contWidget.markerPositionBoxX.valueChanged.connect(self.moveMarkerX)
        self.acquisitionWidget.contWidget.markerPositionBoxY.valueChanged.connect(self.moveMarkerY)
        self.acquisitionWidget.contWidget.markerFactorBox.valueChanged.connect(self.factorChanged)
        self.acquisitionWidget.contWidget.splitButton.toggled.connect(self.splitMarker)
        self.acquisitionWidget.initButton.clicked.connect(self.initializeCamera)
        self.acquisitionWidget.finButton.clicked.connect(self.finalizeCamera)
        self.acquisitionWidget.runButton.toggled.connect(self.startAcquisition)
        self.acquisitionWidget.applyButton.clicked.connect(self.applySettings)

        self.SLM_Controller.pitchBox.valueChanged.connect(self.SLM_pitchChanged)
        self.SLM_Controller.SLMDial.valueChanged.connect(self.SLM_dialChanged)

    def initUI(self):
        # sPTab = self.modeTab.sizePolicy()
        # sPProc = self.processWidget.sizePolicy()
        # sPTab.setVerticalStretch(1)
        # sPProc.setVerticalStretch(1)
        # self.modeTab.setSizePolicy(sPTab)
        # self.processWidget.setSizePolicy(sPProc)
        self.initLayout()

    def initLayout(self):

        vbox0121 = QVBoxLayout()
        vbox0121.addWidget(self.DIOWidget)
        vbox0121.addWidget(self.shutterWidget)

        hbox012 = QHBoxLayout()
        hbox012.addWidget(self.SLM_Controller)
        hbox012.addLayout(vbox0121)
        hbox012.addWidget(self.logWidget)

        vbox01 = QVBoxLayout()
        vbox01.addWidget(self.modeTab)
        vbox01.addWidget(self.processWidget)
        vbox01.addLayout(hbox012)

        hbox0 = QHBoxLayout(self)
        hbox0.addWidget(self.ImgWidget)
        hbox0.addLayout(vbox01)

    def initVal(self):

        self.MarkerX = 0
        self.MarkerY = 0
        self.imgHeight = 600
        self.imgWidth = 600
        self.markerPos = [400, 300]
        self.dst = [0,0,0,0]
        self.split = False
        self.Handle = ct.c_int()
        self.ref = ct.c_int()
        self.CentralPos = None
        self.outDIO = [0, 0, 0, 0, 0, 0]
        self.fin = True

        self.openSettings()

        self.MarkerFactor = self.acquisitionWidget.contWidget.markerFactorBox.value()

        self.applyProcessSettings()

    def initializeCamera(self):
        dll.init(ct.byref(self.Handle))
        self.acquisitionWidget.handleBox.setText(str(self.Handle.value))
        dll.setInitialSettings(self.Handle)
        maxExposeTime = ct.c_double()
        minExposeTime = ct.c_double()
        dll.GetFloatMax(self.Handle, "Exposure Time", ct.byref(maxExposeTime))
        dll.GetFloatMin(self.Handle, "Exposure Time", ct.byref(minExposeTime))
        self.acquisitionWidget.exposeTBox.setRange(minExposeTime.value, maxExposeTime.value)
        logging.info('Minimum exp. time: {}'.format(minExposeTime.value))
        logging.info('Successfully initializied')
        self.acquisitionWidget.initButton.setEnabled(False)
        self.acquisitionWidget.applyButton.setEnabled(True)
        self.acquisitionWidget.finButton.setEnabled(True)

        self.DIOhandle = ct.c_void_p()
        dll.initDIO(ct.byref(self.DIOhandle))
        self.fin = False

    def finalizeCamera(self):
        dll.fin(self.Handle)
        logging.info('Successfully finalized\n')
        self.acquisitionWidget.finButton.setEnabled(False)
        self.acquisitionWidget.runButton.setEnabled(False)
        self.acquisitionWidget.applyButton.setEnabled(False)
        self.acquisitionWidget.initButton.setEnabled(True)
        dll.finDIO(self.DIOhandle)
        self.fin = True

    def moveMarkerX(self, val):
        self.MarkerX = val
        self.calcDst()

    def moveMarkerY(self, val):
        self.MarkerY = val
        self.calcDst()

    def factorChanged(self, val):
        self.MarkerFactor = val
        self.calcDst()

    def SLM_dialChanged(self, val):
        self.SLM_Controller.theta = val/180*np.pi
        self.calcDst()
        if self.SLM_Controller.SLMButton.isChecked():
            if self.SLM_Controller.modulateButton.isChecked():
                self.SLM_Controller.update_img()

    def SLM_pitchChanged(self, val):
        self.SLM_Controller.pitch = val
        self.calcDst()
        if self.SLM_Controller.SLMButton.isChecked():
            if self.SLM_Controller.modulateButton.isChecked():
                self.SLM_Controller.update_img()

    def splitMarker(self, check):
        self.split = check

    def calcDst(self):
        vec = np.array([np.cos(self.SLM_Controller.theta+np.pi/4), np.sin(self.SLM_Controller.theta+np.pi/4)])\
        *self.MarkerFactor/self.SLM_Controller.pitch
        markerPos = np.array([self.imgWidth/2+self.MarkerX, self.imgHeight/2+self.MarkerY])
        self.markerPos = markerPos.astype(np.uint16).tolist()
        rotation = [np.array([[1,0],[0,1]]), np.array([[0,-1],[1,0]]),
                    np.array([[-1,0],[0,-1]]), np.array([[0,1],[-1,0]])]
        for rot, n in zip(rotation, range(4)):
            dst = markerPos + np.dot(rot, vec)
            self.dst[n] = dst.astype(np.uint16).tolist()

    def update_frame(self, source):

        img_height = source.height
        img_width = source.width
        img = source.img
        scale_w = float(800) / float(img_width)
        scale_h = float(600) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        self.imgHeight = height
        self.imgWidth = width
        img = cv2.circle(img, tuple(self.markerPos), 2, (255,0,0), -1)
        if self.split:
            for dst in self.dst:
                img = cv2.circle(img, tuple(dst), 2, (0,255,0), -1)

        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)
        if isinstance(source, ImageAcquirer):
            self.acquisitionWidget.contWidget.imgMaxBox.setText(str(source.max))
            self.acquisitionWidget.contWidget.imgMinBox.setText(str(source.min))


    def startAcquisition(self, checked):
        if checked:
            count = ct.c_int(0)
            count_p = ct.pointer(count)
            if self.acquisitionWidget.Tab.currentIndex() == 0:
                self.applyProcessSettings()
                self.imageAcquirer.setup(self.Handle, fixed=1)
                self.applyProcessSettings()
                logging.info('Acquisition start')
                self.calcDst()
                self.acquisitionWidget.runButton.setText("STOP")
                self.imageAcquirer.start()

            elif self.acquisitionWidget.Tab.currentIndex() == 1:
                if self.acquisitionWidget.fixedWidget.periodButton.isChecked():
                    mainDir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                    num = int(self.acquisitionWidget.fixedWidget.numImgBox.text())
                    repeat = self.acquisitionWidget.fixedWidget.repeatBox.value()
                    self.writeDIO(2)
                    logging.info('Acquisition start')
                    for i in range(repeat*2):
                        self.acquisitionWidget.fixedWidget.currentRepeatBox.setText(str(i%2)+" of "+str(i//2))
                        self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(mainDir),
                                                 num=ct.c_int(num), count_p=count_p)
                        self.imageAcquirer.start()
                        ref = count.value
                        while not self.imageAcquirer.stopped:
                            if count.value > (ref + 5):
                                self.acquisitionWidget.fixedWidget.countBox.setText(str(count.value))
                                QApplication.processEvents()
                                ref = count.value
                        count = ct.c_int(0)
                        count_p = ct.pointer(count)
                        self.writeDIO(3-i%2)
                        waitTime = time.time()
                        while True:
                            now = time.time()
                            if (now - 3) > waitTime:
                                break
                    self.acquisitionWidget.runButton.setChecked(False)
                    logging.info('Acquisition stopped')
                else:
                    dir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                    num = int(self.acquisitionWidget.fixedWidget.numImgBox.text())
                    if self.acquisitionWidget.fixedWidget.aqTypeBox.currentIndex()==0:
                        self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(dir),
                                                 num=ct.c_int(num), count_p=count_p)
                        logging.info('Acquisition start')
                        self.imageAcquirer.start()
                        while not self.imageAcquirer.stopped:
                            QTimer.singleShot(30, lambda: self.acquisitionWidget.fixedWidget.countBox.setText(str(count.value)))
                            QApplication.processEvents()
                        logging.info('Acquisition stopped\n')
                        self.acquisitionWidget.runButton.setChecked(False)
                    elif self.acquisitionWidget.fixedWidget.aqTypeBox.currentIndex()==1:
                        self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(dir),
                                                 num=ct.c_int(num), count_p=count_p,
                                                 center=self.CentralPos, DIOthres=self.acquisitionWidget.fixedWidget.thresBox.value(),
                                                 DIOhandle=self.DIOhandle, mode=1)
                        self.applyProcessSettings()
                        self.imageAcquirer.start()
                    elif self.acquisitionWidget.fixedWidget.aqTypeBox.currentIndex()==2:
                        self.imageAcquirer.setup(self.Handle,  dir=ct.c_char_p(dir),
                                                 num=ct.c_int(num), count_p=count_p,
                                                 center=self.CentralPos, DIOthres=self.acquisitionWidget.fixedWidget.thresBox.value(),
                                                 DIOhandle=self.DIOhandle, mode=2)
                        self.applyProcessSettings()
                        self.imageAcquirer.start()
            # else:
            #     if self.imageLoader.dirname:
            #         self.applyProcessSettings()
            #         self.imagePlayer.setup()
            #         self.acquisitionWidget.runButton.setText('STOP')
            #         self.imagePlayer.run()
            #     else:
            #         logging.error('No directory selected!')
            #         self.acquisitionWidget.runButton.setChecked(False)

        else:
            self.imageAcquirer.stopped = True
            self.imagePlayer.stopped = True
            self.acquisitionWidget.runButton.setText("RUN")

    def acquisitionFinished(self):
        self.acquisitionWidget.runButton.setChecked(False)

    def applyCenter(self, posX, posY):
        self.CentralPos = (posX, posY)
        self.acquisitionWidget.runButton.setChecked(False)
        self.acquisitionWidget.fixedWidget.cntrPosLabel.setText("Center of position: ({:.2f}, {:.2f})".format(self.CentralPos[0], self.CentralPos[1]))

    def setAOICenter(self):
        centerX = self.acquisitionWidget.CenterXBox.value()
        centerY = self.acquisitionWidget.CenterYBox.value()
        AOISizeIndex = self.acquisitionWidget.AOISizeBox.currentIndex()
        AOISize = 2048/2**AOISizeIndex
        if dll.SetInt(self.Handle, "AOIWidth", int(AOISize)):
            logging.error("AOIWidth")
        if dll.SetInt(self.Handle, "AOILeft", int(1024 + centerX - AOISize/2)):
            logging.error("AOILeft")
        if dll.SetInt(self.Handle, "AOIHeight", int(AOISize)):
            logging.error("AOIHeight")
        if dll.SetInt(self.Handle, "AOITop", int(1024 + centerY - AOISize/2)):
            logging.error("AOILeft")

    def applySettings(self):
        dll.SetFloat(self.Handle, "Exposure Time", self.acquisitionWidget.exposeTBox.value())
        dll.SetEnumString(self.Handle, "AOIBinning", self.acquisitionWidget.AOIBinBox.currentText())
        index = self.acquisitionWidget.AOISizeBox.currentIndex()
        if index == 5:
            if dll.SetInt(self.Handle, "AOIWidth", self.acquisitionWidget.AOIWidthBox.value()):
                logging.error("AOIWidth")
            if dll.SetInt(self.Handle, "AOILeft", self.acquisitionWidget.AOILeftBox.value()):
                logging.error("AOILeft")
            if dll.SetInt(self.Handle, "AOIHeight", self.acquisitionWidget.AOIHeightBox.value()):
                logging.error("AOIHeight")
            if dll.SetInt(self.Handle, "AOITop", self.acquisitionWidget.AOITopBox.value()):
                logging.error("AOITop")
        elif index == 0:
            if dll.SetInt(self.Handle, "AOIWidth", 2048):
                logging.error("AOIWidth")
            if dll.SetInt(self.Handle, "AOIHeight", 2048):
                logging.error("AOIHeight")
        else:
            self.setAOICenter()
        # else:
        #     if dll.SetInt(self.Handle, "AOIWidth", int(2048/(2**self.acquisitionWidget.AOISizeBox.currentIndex()))):
        #         logging.error("AOIWidth")
        #     if dll.SetInt(self.Handle, "AOIHeight", int(2048/(2**self.acquisitionWidget.AOISizeBox.currentIndex()))):
        #         logging.error("AOIHeight")
        #     dll.centreAOI(self.Handle)
        dll.SetBool(self.Handle, "RollingShutterGlobalClear", self.acquisitionWidget.globalClearButton.isChecked())

        frameRate = ct.c_double(float(self.acquisitionWidget.fixedWidget.frameRateBox.text()))
        frameRateMax = ct.c_double()
        dll.GetFloatMax(self.Handle, "FrameRate", ct.byref(frameRateMax))
        logging.info("Max Framerate: {:.2f}".format(frameRateMax.value))
        if (frameRate.value > frameRateMax.value):
            frameRate = frameRateMax
            self.acquisitionWidget.fixedWidget.frameRateBox.setText(str(frameRate.value))
            logging.warning("Too fast frame rate! You'd better to change!")
        dll.SetFloat(self.Handle, ct.c_wchar_p("FrameRate"), frameRate)
        logging.info("Settings applied")
        self.acquisitionWidget.runButton.setEnabled(True)

    def applyProcessClicked(self):
        if self.modeTab.currentIndex() == 1:
            self.applyProcessSettings(True)
        else:
            self.applyProcessSettings(False)

    def applyProcessSettings(self, update=False):
        self.processWidget.set_prm()
        self.imageAcquirer.prms = self.processWidget.prmStruct
        self.imageLoader.prms = self.processWidget.prmStruct
        if update:
            self.imageLoader.update_img()

    def writeDIO(self, signal):
        # select port from signal
        port = signal // 2
        # take value (0 or 1) from signal
        val = signal % 2
        # write value into the index corresponding to port of list
        self.outDIO[-1*port-1] = val
        # list to strings
        strings = ''.join(map(str, self.outDIO))
        out = int(strings, 2)
        logging.debug(strings)
        dll.writeDIO(self.DIOhandle, ct.c_ubyte(out))

    def analyzeDatas(self):
        self.applyProcessSettings()
        logging.info("applySettings")
        start = self.imageLoader.anlzStartBox.value()
        end = self.imageLoader.anlzEndBox.value()
        processfiles = []
        for i in range(start, end):
            processfiles.append(self.imageLoader.datfiles[i])
        logging.info("setup")
        self.imageProcesser.setup(processfiles, self.imageLoader.width,
                                  self.imageLoader.height, self.imageLoader.stride,
                                  self.processWidget.prmStruct, self.imageLoader.dirname, self.imageLoader.progressBar)
        self.imageProcesser.run()

    def exportBMP(self):
        fileToSave = QFileDialog.getSaveFileName(self, 'File to save')
        logging.info(fileToSave)
        if fileToSave:
            cv2.imwrite(fileToSave[0], self.imageLoader.img)

    def exportSeries(self):
        dirToSave = QFileDialog.getExistingDirectory(self, 'Select directory')
        logging.debug(dirToSave)
        if dirToSave:
            start = self.imageLoader.anlzStartBox.value()
            end = self.imageLoader.anlzEndBox.value()
            for i in range(start, end):
                rawdata = np.fromfile(self.imageLoader.datfiles[i], dtype=np.uint8)
                buffer = rawdata.ctypes.data_as(ct.POINTER(ct.c_ubyte))
                outputBuffer = (ct.c_ushort * (self.imageLoader.width * self.imageLoader.height))()
                outBuffer = (ct.c_ubyte*(self.imageLoader.width*self.imageLoader.height))()
                max = ct.c_double()
                min = ct.c_double()
                ret = dll.convertBuffer(buffer, outputBuffer, self.imageLoader.width, self.imageLoader.height, self.imageLoader.stride)
                dll.processImageShow(self.imageLoader.height, self.imageLoader.width, outputBuffer, self.imageLoader.prms, outBuffer,
                                     ct.byref(max), ct.byref(min))
                img = np.array(outBuffer).reshape(self.imageLoader.height, self.imageLoader.width)
                cv2.imwrite(dirToSave+"/"+str(i)+".bmp", img)


    def openSettings(self):
        self.settings = QSettings('setting.ini', 'Andor_GUI')
        settings = self.settings
        self.acquisitionWidget.fixedWidget.dirname = settings.value('dirname', '')
        self.acquisitionWidget.fixedWidget.dirBox.setText(self.acquisitionWidget.fixedWidget.dirname)
        self.acquisitionWidget.Tab.setCurrentIndex(settings.value('Cycle', type=int))
        self.processWidget.blurButton.setChecked(settings.value('blur', type=bool))
        self.processWidget.blurSize.setValue(settings.value('blur size', type=int))
        self.processWidget.blurSigma.setValue(settings.value('blur sigma', type=int))
        self.processWidget.thresButton.setChecked(settings.value('thres', type=bool))
        self.processWidget.thresVal.setValue(settings.value('thres val', type=int))
        self.processWidget.thresType.setCurrentIndex(settings.value('thres type', type=int))
        self.processWidget.OtsuButton.setChecked(settings.value('thres otsu', type=bool))
        self.processWidget.contButton.setChecked(settings.value('cont', type=bool))
        self.processWidget.contNum.setValue(settings.value('cont num', type=int))
        self.acquisitionWidget.exposeTBox.setValue(settings.value('exposure time', 0.01, type=float))
        self.acquisitionWidget.AOILeftBox.setValue(settings.value('AOI left', type=int))
        self.acquisitionWidget.AOITopBox.setValue(settings.value('AOI top', type=int))
        self.acquisitionWidget.AOIWidthBox.setValue(settings.value('AOI width', type=int))
        self.acquisitionWidget.AOIHeightBox.setValue(settings.value('AOI height', type=int))
        self.acquisitionWidget.CenterXBox.setValue(settings.value('Center X', type=int))
        self.acquisitionWidget.CenterYBox.setValue(settings.value('Center Y', type=int))
        self.acquisitionWidget.fixedWidget.thresBox.setValue(settings.value('DIO threshold', type=float))
        self.acquisitionWidget.contWidget.markerFactorBox.setValue(settings.value('marker factor', 100, type=float))
        self.imageLoader.dirname = settings.value('dir image', '')
        self.SLM_Controller.wavelengthBox.setCurrentIndex(settings.value('SLM wavelength', type=int))
        self.SLM_Controller.pitchBox.setValue(settings.value('SLM pitch', 23, type=int))
        self.SLM_Controller.focusBox.setValue(settings.value('SLM focus', 0, type=float))
        self.SLM_Controller.focusXBox.setValue(settings.value('SLM focusX', 0, type=int))
        self.SLM_Controller.focusYBox.setValue(settings.value('SLM focusY', 0, type=int))

    def writeSettings(self):  # Save current settings
        self.settings = QSettings('setting.ini', 'Andor_GUI')
        self.settings.setValue('dirname', self.acquisitionWidget.fixedWidget.dirname)
        self.settings.setValue('Cycle', self.acquisitionWidget.Tab.currentIndex())
        self.settings.setValue('blur', self.processWidget.blurButton.isChecked())
        self.settings.setValue('blur size', self.processWidget.blurSize.value())
        self.settings.setValue('blur sigma', self.processWidget.blurSigma.value())
        self.settings.setValue('thres', self.processWidget.thresButton.isChecked())
        self.settings.setValue('thres val', self.processWidget.thresVal.value())
        self.settings.setValue('thres type', self.processWidget.thresType.currentIndex())
        self.settings.setValue('thres otsu', self.processWidget.OtsuButton.isChecked())
        self.settings.setValue('cont', self.processWidget.contButton.isChecked())
        self.settings.setValue('cont num', self.processWidget.contNum.value())
        self.settings.setValue('exposure time', self.acquisitionWidget.exposeTBox.value())
        self.settings.setValue('AOI left', self.acquisitionWidget.AOILeftBox.value())
        self.settings.setValue('AOI top', self.acquisitionWidget.AOITopBox.value())
        self.settings.setValue('AOI width', self.acquisitionWidget.AOIWidthBox.value())
        self.settings.setValue('AOI height', self.acquisitionWidget.AOIHeightBox.value())
        self.settings.setValue('Center X', self.acquisitionWidget.CenterXBox.value())
        self.settings.setValue('Center Y', self.acquisitionWidget.CenterYBox.value())
        self.settings.setValue('DIO threshold', self.acquisitionWidget.fixedWidget.thresBox.value())
        self.settings.setValue('marker factor', self.acquisitionWidget.contWidget.markerFactorBox.value())
        self.settings.setValue('dir image', self.imageLoader.dirname)
        self.settings.setValue('SLM wavelength', self.SLM_Controller.wavelengthBox.currentIndex())
        self.settings.setValue('SLM pitch', self.SLM_Controller.pitchBox.value())
        self.settings.setValue('SLM focus', self.SLM_Controller.focusBox.value())
        self.settings.setValue('SLM focusX', self.SLM_Controller.focusXBox.value())
        self.settings.setValue('SLM focusY', self.SLM_Controller.focusYBox.value())


class mainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)

        menu = self.menuBar()
        fileMenu = menu.addMenu('File')

        self.toolbar = self.addToolBar('Toolbar')

        self.central = centralWidget(self)
        self.setCentralWidget(self.central)

        expAct = QAction('Export img', self)
        expAct.triggered.connect(self.central.exportBMP)
        expSAct = QAction('Export image series', self)
        expSAct.triggered.connect(self.central.exportSeries)
        exitAct = QAction('Exit', self)
        exitAct.triggered.connect(self.close)
        fileMenu.addAction(expAct)
        fileMenu.addAction(expSAct)
        fileMenu.addAction(exitAct)

        # self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle('Andor_CMOS')

        self.show()

    def closeEvent(self, event):
        if not self.central.fin:
                self.central.finalizeCamera()
        self.central.writeSettings()
        dll.FinaliseLibrary()
        dll.FinaliseUtilityLibrary()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainWindow()
    window.move(20, 20)
    sys.exit(app.exec_())
