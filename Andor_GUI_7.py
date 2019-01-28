# -*- coding: utf-8 -*-

import ctypes as ct
import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
from glob import glob
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir, Qt, QSettings, QTimer
from PyQt5 import QtGui
from PyQt5 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg\
 import FigureCanvasQTAgg as FigureCanvas

# os.chdir(os.path.dirname(__file__)+r'\Andor_dll\Andor_dll')
os.chdir(r'Andor_dll\Andor_dll')


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
    _fields_ = [("blur", blurPrms),
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
        mean = ct.c_double()
        std = ct.c_double()
        while not self.stopped:
            if dll.WaitBuffer(self.Handle, ct.byref(Buffer), ct.byref(BufferSize), 10000) == 0:
                ret = dll.convertBuffer(Buffer, outputBuffer, ImageWidth, ImageHeight, ImageStride)
                dll.processImageShow(ImageHeight, ImageWidth, outputBuffer, self.prms,
                                     outBuffer, ct.byref(mean), ct.byref(std))
                self.mean = mean.value
                self.std = std.value
                self.img = np.array(outBuffer).reshape(ImageHeight.value, ImageWidth.value)
                self.width = ImageWidth.value
                self.height = ImageHeight.value
                self.imgSignal.emit(self)
                dll.QueueBuffer(self.Handle, Buffer, BufferSize)
            else:
                print('failed')
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

    def setup(self, datfiles, width, height, stride, prms, dir):
        self.datfiles = datfiles
        self.width = width
        self.height = height
        self.stride = stride
        self.prms = prms
        self.dir = dir
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
        for datfile in self.datfiles:
            rawdata = np.fromfile(datfile, dtype=np.uint8)
            buffer = rawdata.ctypes.data_as(ct.POINTER(ct.c_ubyte))
            ret = dll.convertBuffer(buffer, ImgArry, self.width, self.height, self.stride)
            dll.processImage(point, self.height, self.width, ImgArry, self.prms)
            pointlst.append([point[0], point[1]])
        DF = pd.DataFrame(np.array(pointlst))
        DF.columns = ['x', 'y']
        DF.to_csv(self.dir + r"\COG.csv")
        print("analysis finished")


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


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None
        self.setMinimumSize(600, 600)

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


class contWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.imgMaxBox = QLineEdit(self)
        self.imgMinBox = QLineEdit(self)
        self.markerPositionBoxX = QSpinBox(self)
        self.markerPositionBoxX.setMaximum(400)
        self.markerPositionBoxX.setMinimum(-400)
        self.markerPositionBoxY = QSpinBox(self)
        self.markerPositionBoxY.setMaximum(400)
        self.markerPositionBoxY.setMinimum(-400)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        vlayout = QVBoxLayout(self)

        hbox1.addWidget(QLabel('Max'))
        hbox1.addWidget(self.imgMaxBox)
        hbox1.addWidget(QLabel('Min'))
        hbox1.addWidget(self.imgMinBox)
        hbox2.addWidget(QLabel('x: '), alignment=Qt.AlignRight)
        hbox2.addWidget(self.markerPositionBoxX)
        hbox2.addWidget(QLabel('y: '), alignment=Qt.AlignRight)
        hbox2.addWidget(self.markerPositionBoxY)
        vlayout.addLayout(hbox1)
        vlayout.addWidget(QLabel('Marker position'))
        vlayout.addLayout(hbox2)


class fixedWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        numImgBox = QLineEdit('100', self)
        frameRateBox = QLineEdit('30', self)
        countBox = QLineEdit(self)
        dirBox = QLineEdit(self)
        dirBox.setReadOnly(True)
        dirButton = QPushButton('...', self)
        dirButton.clicked.connect(self.selectDirectory)
        self.aqTypeBox = QComboBox(self)
        aqTypeList = ['No feed-back', 'Preparation', 'Do feed-back']
        self.aqTypeBox.addItems(aqTypeList)
        self.cntrPosLabel = QLabel("Centre of position:", self)
        self.thresBox = QDoubleSpinBox(self)

        vbox = QVBoxLayout(self)
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        hbox1.addWidget(QLabel('Number of frames', self))
        hbox1.addWidget(numImgBox)
        hbox1.addWidget(QLabel('Current frame', self))
        hbox1.addWidget(countBox)
        hbox1.addWidget(QLabel('FrameRate', self))
        hbox1.addWidget(frameRateBox)
        hbox1.addWidget(QLabel('Directory', self))
        hbox1.addWidget(dirBox)
        hbox1.addWidget(dirButton)

        hbox2.addWidget(self.aqTypeBox)
        hbox2.addWidget(self.cntrPosLabel)
        hbox2.addWidget(QLabel("Threshold:"), alignment=Qt.AlignRight)
        hbox2.addWidget(self.thresBox)

        self.numImgBox = numImgBox
        self.countBox = countBox
        self.frameRateBox = frameRateBox
        self.dirBox = dirBox

    def selectDirectory(self):
        self.dirname = QFileDialog.getExistingDirectory(self, 'Select directory', self.dirname)
        self.dirname = QDir.toNativeSeparators(self.dirname)
        self.dirBox.setText(self.dirname)


class imageLoader(QWidget):

    imgSignal = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dirname = None
        self.img = None
        self.parent = parent

        self.dirBox = QLineEdit(self)
        self.dirBox.setReadOnly(True)
        self.dirButton = QPushButton('...', self)
        self.dirButton.clicked.connect(self.selectDirectory)
        self.fileNumBox = QLineEdit(self)
        self.fileNumBox.setReadOnly(True)
        self.currentNumBox = QSpinBox(self)
        self.currentNumBox.valueChanged.connect(self.update_img)
        self.currentNumBox.setWrapping(True)

        self.anlzButton = QPushButton('Analysis', self)
        self.anlzStartBox = QSpinBox(self)
        self.anlzEndBox = QSpinBox(self)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel('Directory:'), alignment=Qt.AlignRight)
        hbox1.addWidget(self.dirBox)
        hbox1.addWidget(self.dirButton)
        hbox1.addWidget(QLabel('Number:'), alignment=Qt.AlignRight)
        hbox1.addWidget(self.fileNumBox)
        hbox1.addWidget(QLabel('Current:'), alignment=Qt.AlignRight)
        hbox1.addWidget(self.currentNumBox)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.anlzButton)
        hbox2.addWidget(QLabel('Start: '), alignment=Qt.AlignRight)
        hbox2.addWidget(self.anlzStartBox)
        hbox2.addWidget(QLabel('End: '), alignment=Qt.AlignRight)
        hbox2.addWidget(self.anlzEndBox)

        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

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
            print('No metadata was found.\n')

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
        mean = ct.c_double()
        std = ct.c_double()
        ret = dll.convertBuffer(buffer, outputBuffer, self.width, self.height, self.stride)
        dll.processImageShow(self.height, self.width, outputBuffer, self.prms, outBuffer,
                             ct.byref(mean), ct.byref(std))
        self.mean = mean.value
        self.std = std.value
        self.img = np.array(outBuffer).reshape(self.height, self.width)
        self.imgSignal.emit(self)


class processWidget(QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.prmStruct = processPrms(blurPrms(0,0,0,0),
                                     thresPrms(0,0,0,0),
                                     contPrms(0,0))

        layout = QVBoxLayout(self)

        layout_blur = QHBoxLayout()
        self.blurButton = QRadioButton('Blur', self)
        self.blurButton.setAutoExclusive(False)
        self.blurSize = QSpinBox(self)
        self.blurSigma = QSpinBox(self)
        layout_blur.addWidget(self.blurButton)
        layout_blur.addWidget(QLabel('Kernel size : '), alignment=Qt.AlignRight)
        layout_blur.addWidget(self.blurSize)
        layout_blur.addWidget(QLabel('Sigma : '), alignment=Qt.AlignRight)
        layout_blur.addWidget(self.blurSigma)

        layout_thres = QHBoxLayout(self)
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
        layout_thres.addWidget(self.thresButton)
        layout_thres.addWidget(QLabel('Value : '), alignment=Qt.AlignRight)
        layout_thres.addWidget(self.thresVal)
        layout_thres.addWidget(QLabel('Type : '), alignment=Qt.AlignRight)
        layout_thres.addWidget(self.thresType)
        layout_thres.addWidget(self.OtsuButton, alignment=Qt.AlignCenter)

        layout_cont = QHBoxLayout(self)
        self.contButton = QRadioButton('Find Contours', self)
        self.contButton.setAutoExclusive(False)
        self.contNum = QSpinBox(self)
        self.contNum.setRange(1,4)
        layout_cont.addWidget(self.contButton)
        layout_cont.addWidget(QLabel('Number to find : '), alignment=Qt.AlignRight)
        layout_cont.addWidget(self.contNum)

        self.applyButton = QPushButton('APPLY')

        layout.addWidget(QLabel("Process settings"))
        layout.addStretch(1)
        layout.addLayout(layout_blur)
        layout.addStretch(1)
        layout.addLayout(layout_thres)
        layout.addStretch(1)
        layout.addLayout(layout_cont)
        layout.addStretch(1)
        layout.addWidget(self.applyButton)

        self.setStyleSheet("background-color:white;")

    def set_prm(self):
        self.prmStruct.blur = blurPrms(self.blurButton.isChecked(), self.blurSize.value(), self.blurSize.value(), self.blurSigma.value())
        self.prmStruct.thres = thresPrms(self.thresButton.isChecked(), self.thresVal.value(), 255, self.thresType.currentIndex()+int(self.OtsuButton.isChecked())*8)
        self.prmStruct.cont = contPrms(self.contButton.isChecked(), self.contNum.value())


class DIOWidget(QFrame):
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
        self.DIObuttons.setId(self.DIOLowButton, 0)
        self.DIObuttons.setId(self.DIOHighButton, 1)

        vbox = QVBoxLayout(self)
        vbox.addWidget(QLabel('DIO Control'))
        vbox.addWidget(self.DIOHighButton)
        vbox.addWidget(self.DIOLowButton)

        self.DIOLowButton.setChecked(True)


class centralWidget(QWidget):

    def __init__(self, parent=None):
        super(centralWidget, self).__init__(parent)

        self.initUI()

        self.imageAcquirer = ImageAcquirer()
        self.imageAcquirer.imgSignal.connect(self.update_frame)
        self.imageAcquirer.posSignal.connect(self.applyCenter)
        self.imageAcquirer.finished.connect(self.acquisitionFinished)

        self.imagePlayer = ImagePlayer()
        self.imagePlayer.countStepSignal.connect(self.imageLoader.currentNumBox.stepUp)

        self.imageProcesser = ImageProcesser()

        self.MarkerX = 0
        self.MarkerY = 0

        self.Handle = ct.c_int()
        self.ref = ct.c_int()
        self.openSettings()
        self.CentralPos = None
        self.fin = True

        self.applyProcessSettings()

    def initUI(self):
        self.handleBox = QLineEdit(self)
        self.handleBox.setReadOnly(True)
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
        self.AOIBinBox = QComboBox(self)
        AOIBinList = ["1x1", "2x2", "3x3", "4x4", "8x8"]
        self.AOIBinBox.addItems(AOIBinList)
        self.globalClearButton = QPushButton('Global clear', self)
        self.globalClearButton.setCheckable(True)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel('Handle', self))
        hbox1.addWidget(self.handleBox)
        hbox1.addWidget(QLabel('Exposure time (s)', self))
        hbox1.addWidget(self.exposeTBox)
        hbox1.addWidget(QLabel('AOI Size', self))
        hbox1.addWidget(self.AOISizeBox)
        hbox1.addWidget(QLabel('Top'))
        hbox1.addWidget(self.AOITopBox)
        hbox1.addWidget(QLabel('Left'))
        hbox1.addWidget(self.AOILeftBox)
        hbox1.addWidget(QLabel('Width'))
        hbox1.addWidget(self.AOIWidthBox)
        hbox1.addWidget(QLabel('Height'))
        hbox1.addWidget(self.AOIHeightBox)
        hbox1.addWidget(self.AOIBinBox)
        hbox1.addWidget(self.globalClearButton)
        # hbox1.addWidget(QLabel('AOI Width', self))
        # hbox1.addWidget(self.AOIWidthBox)
        # hbox1.addWidget(QLabel('AOI Height', self))
        # hbox1.addWidget(self.AOIHeightBox)

        self.ImgWidget = OwnImageWidget(self)

        self.contWidget = contWidget(self)
        self.contWidget.markerPositionBoxX.valueChanged.connect(self.moveMarkerX)
        self.contWidget.markerPositionBoxY.valueChanged.connect(self.moveMarkerY)

        self.fixedWidget = fixedWidget(self)

        self.imageLoader = imageLoader(self)
        self.imageLoader.imgSignal.connect(self.update_frame)
        self.imageLoader.anlzButton.clicked.connect(self.analyzeDatas)

        self.processWidget = processWidget(self)
        self.processWidget.applyButton.clicked.connect(self.applyProcessSettings)

        self.DIOWidget = DIOWidget(self)
        self.DIOWidget.DIObuttons.buttonClicked[int].connect(self.writeDIO)

        self.Tab = QTabWidget()
        self.Tab.addTab(self.contWidget, 'Continuous acquisition')
        self.Tab.addTab(self.fixedWidget, 'Fixed acquisition')
        self.Tab.addTab(self.imageLoader, 'Image loader')

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.Tab)
        hbox3.addWidget(self.DIOWidget)

        vbox2 = QVBoxLayout()
        sPTab = self.Tab.sizePolicy()
        sPProc = self.processWidget.sizePolicy()
        sPTab.setVerticalStretch(1)
        sPProc.setVerticalStretch(1)
        self.Tab.setSizePolicy(sPTab)
        self.processWidget.setSizePolicy(sPProc)
        vbox2.addLayout(hbox3)
        vbox2.addWidget(self.processWidget)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.ImgWidget)
        hbox2.addLayout(vbox2)

        self.initButton = QPushButton("Initialize", self)
        self.initButton.clicked.connect(self.initializeCamera)
        self.finButton = QPushButton("Finalize", self)
        self.finButton.clicked.connect(self.finalizeCamera)
        self.finButton.setEnabled(False)
        self.runButton = QPushButton('RUN', self)
        self.runButton.setCheckable(True)
        self.runButton.setEnabled(False)
        self.runButton.toggled.connect(self.startAcquisition)
        self.applyButton = QPushButton('APPLY', self)
        self.applyButton.clicked.connect(self.applySettings)
        self.applyButton.setEnabled(False)
        hbox = QHBoxLayout()
        hbox.addWidget(self.initButton)
        hbox.addWidget(self.applyButton)
        hbox.addWidget(self.runButton)
        hbox.addWidget(self.finButton)

        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def initializeCamera(self):
        dll.init(ct.byref(self.Handle))
        self.handleBox.setText(str(self.Handle.value))
        dll.setInitialSettings(self.Handle)
        maxExposeTime = ct.c_double()
        minExposeTime = ct.c_double()
        dll.GetFloatMax(self.Handle, "Exposure Time", ct.byref(maxExposeTime))
        dll.GetFloatMin(self.Handle, "Exposure Time", ct.byref(minExposeTime))
        self.exposeTBox.setRange(minExposeTime.value, maxExposeTime.value)
        print(minExposeTime.value)
        print('Successfully initializied')
        self.initButton.setEnabled(False)
        self.applyButton.setEnabled(True)
        self.finButton.setEnabled(True)

        self.DIOhandle = ct.c_void_p()
        dll.initDIO(ct.byref(self.DIOhandle))
        self.fin = False

    def finalizeCamera(self):
        dll.fin(self.Handle)
        print('Successfully finalized\n')
        self.finButton.setEnabled(False)
        self.runButton.setEnabled(False)
        self.applyButton.setEnabled(False)
        self.initButton.setEnabled(True)
        dll.finDIO(self.DIOhandle)
        self.fin = True

    def moveMarkerX(self, val):
        self.MarkerX = val

    def moveMarkerY(self, val):
        self.MarkerY = val

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
        img = cv2.circle(img, (width//2 + self.MarkerX, height//2 + self.MarkerY), 2, (255,0,0), -1)
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)
        if isinstance(source, ImageAcquirer):
            self.contWidget.imgMaxBox.setText(str(source.mean+2*source.std))
            self.contWidget.imgMinBox.setText(str(source.mean-2*source.std))


    def startAcquisition(self, checked):
        if checked:
            count = ct.c_int(0)
            count_p = ct.pointer(count)
            if self.Tab.currentIndex() == 0:
                self.applyProcessSettings()
                self.imageAcquirer.setup(self.Handle, fixed=1)
                self.applyProcessSettings()
                print('Acquisition start')
                self.runButton.setText("STOP")
                self.imageAcquirer.start()

            elif self.Tab.currentIndex() == 1:
                dir = self.fixedWidget.dirname.encode(encoding='utf_8')
                num = int(self.fixedWidget.numImgBox.text())
                if self.fixedWidget.aqTypeBox.currentIndex()==0:
                    self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(dir),
                                             num=ct.c_int(num), count_p=count_p)
                    print('Acquisition start')
                    self.imageAcquirer.start()
                    while not self.imageAcquirer.stopped:
                        QTimer.singleShot(30, lambda: self.fixedWidget.countBox.setText(str(count.value)))
                        QApplication.processEvents()
                    print('Acquisition stopped\n')
                    self.runButton.setChecked(False)
                elif self.fixedWidget.aqTypeBox.currentIndex()==1:
                    self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(dir),
                                             num=ct.c_int(num), count_p=count_p,
                                             center=self.CentralPos, DIOthres=self.fixedWidget.thresBox.value(),
                                             DIOhandle=self.DIOhandle, mode=1)
                    self.applyProcessSettings()
                    self.imageAcquirer.start()
                elif self.fixedWidget.aqTypeBox.currentIndex()==2:
                    self.imageAcquirer.setup(self.Handle,  dir=ct.c_char_p(dir),
                                             num=ct.c_int(num), count_p=count_p,
                                             center=self.CentralPos, DIOthres=self.fixedWidget.thresBox.value(),
                                             DIOhandle=self.DIOhandle, mode=2)
                    self.applyProcessSettings()
                    self.imageAcquirer.start()
            else:
                if self.imageLoader.dirname:
                    self.applyProcessSettings()
                    self.imagePlayer.setup()
                    self.runButton.setText('STOP')
                    self.imagePlayer.run()
                else:
                    print('ERROR: No directory selected!')
                    self.runButton.setChecked(False)

        else:
            self.imageAcquirer.stopped = True
            self.imagePlayer.stopped = True
            self.runButton.setText("RUN")

    def acquisitionFinished(self):
        self.runButton.setChecked(False)

    def applyCenter(self, posX, posY):
        self.CentralPos = (posX, posY)
        self.runButton.setChecked(False)
        self.fixedWidget.cntrPosLabel.setText("Center of position: ({:.2f}, {:.2f})".format(self.CentralPos[0], self.CentralPos[1]))


    def applySettings(self):
        dll.SetFloat(self.Handle, "Exposure Time", self.exposeTBox.value())
        dll.SetEnumString(self.Handle, "AOIBinning", self.AOIBinBox.currentText())
        if self.AOISizeBox.currentIndex() == 5:
            if dll.SetInt(self.Handle, "AOIWidth", self.AOIWidthBox.value()):
                print("ERROR: AOIWidth")
            if dll.SetInt(self.Handle, "AOILeft", self.AOILeftBox.value()):
                print("ERROR: AOILeft")
            if dll.SetInt(self.Handle, "AOIHeight", self.AOIHeightBox.value()):
                print("ERROR: AOIHeight")
            if dll.SetInt(self.Handle, "AOITop", self.AOILeftBox.value()):
                print("ERROR: AOITop")
        else:
            if dll.SetInt(self.Handle, "AOIWidth", int(2048/(2**self.AOISizeBox.currentIndex()))):
                print("ERROR: AOIWidth")
            if dll.SetInt(self.Handle, "AOIHeight", int(2048/(2**self.AOISizeBox.currentIndex()))):
                print("ERROR: AOIHeight")
            dll.centreAOI(self.Handle)
        dll.SetBool(self.Handle, "RollingShutterGlobalClear", self.globalClearButton.isChecked())

        frameRate = ct.c_double(float(self.fixedWidget.frameRateBox.text()))
        frameRateMax = ct.c_double()
        dll.GetFloatMax(self.Handle, "FrameRate", ct.byref(frameRateMax))
        print("Max Framerate: ", frameRateMax.value)
        if (frameRate.value > frameRateMax.value):
            frameRate = frameRateMax
            self.fixedWidget.frameRateBox.setText(str(frameRate.value))
            print("Too fast frame rate! You'd better to change!")
        dll.SetFloat(self.Handle, ct.c_wchar_p("FrameRate"), frameRate)
        print("Settings applied\n")
        self.runButton.setEnabled(True)

    def applyProcessSettings(self):
        self.processWidget.set_prm()
        self.imageAcquirer.prms = self.processWidget.prmStruct
        self.imageLoader.prms = self.processWidget.prmStruct

    def writeDIO(self, val):
        dll.writeDIO(self.DIOhandle, ct.c_ubyte(val))

    def analyzeDatas(self):
        self.applyProcessSettings()
        print("applySettings")
        start = self.imageLoader.anlzStartBox.value()
        end = self.imageLoader.anlzEndBox.value()
        processfiles = []
        for i in range(start, end):
            processfiles.append(self.imageLoader.datfiles[i])
        print("setup")
        self.imageProcesser.setup(processfiles, self.imageLoader.width,
                                  self.imageLoader.height, self.imageLoader.stride,
                                  self.processWidget.prmStruct, self.imageLoader.dirname)
        self.imageProcesser.run()

    def exportBMP(self):
        fileToSave = QFileDialog.getSaveFileName(self, 'File to save')
        print(fileToSave)
        if fileToSave:
            cv2.imwrite(fileToSave[0], self.imageLoader.img)

    def openSettings(self):
        self.settings = QSettings('setting.ini', 'Andor_GUI')
        settings = self.settings
        self.fixedWidget.dirname = settings.value('dirname', '')
        self.fixedWidget.dirBox.setText(self.fixedWidget.dirname)
        self.Tab.setCurrentIndex(settings.value('Cycle', type=int))
        self.processWidget.blurButton.setChecked(settings.value('blur', type=bool))
        self.processWidget.blurSize.setValue(settings.value('blur size', type=int))
        self.processWidget.blurSigma.setValue(settings.value('blur sigma', type=int))
        self.processWidget.thresButton.setChecked(settings.value('thres', type=bool))
        self.processWidget.thresVal.setValue(settings.value('thres val', type=int))
        self.processWidget.thresType.setCurrentIndex(settings.value('thres type', type=int))
        self.processWidget.OtsuButton.setChecked(settings.value('thres otsu', type=bool))
        self.processWidget.contButton.setChecked(settings.value('cont', type=bool))
        self.processWidget.contNum.setValue(settings.value('cont num', type=int))
        self.exposeTBox.setValue(settings.value('exposure time', 0.01, type=float))
        self.fixedWidget.thresBox.setValue(settings.value('DIO threshold', type=float))
        self.imageLoader.dirname = settings.value('dir image', '')

    def writeSettings(self):  # Save current settings
        self.settings = QSettings('setting.ini', 'Andor_GUI')
        self.settings.setValue('dirname', self.fixedWidget.dirname)
        self.settings.setValue('Cycle', self.Tab.currentIndex())
        self.settings.setValue('blur', self.processWidget.blurButton.isChecked())
        self.settings.setValue('blur size', self.processWidget.blurSize.value())
        self.settings.setValue('blur sigma', self.processWidget.blurSigma.value())
        self.settings.setValue('thres', self.processWidget.thresButton.isChecked())
        self.settings.setValue('thres val', self.processWidget.thresVal.value())
        self.settings.setValue('thres type', self.processWidget.thresType.currentIndex())
        self.settings.setValue('thres otsu', self.processWidget.OtsuButton.isChecked())
        self.settings.setValue('cont', self.processWidget.contButton.isChecked())
        self.settings.setValue('cont num', self.processWidget.contNum.value())
        self.settings.setValue('exposure time', self.exposeTBox.value())
        self.settings.setValue('DIO threshold', self.fixedWidget.thresBox.value())
        self.settings.setValue('dir image', self.imageLoader.dirname)

class Logger(object):
    def __init__(self, editor, out=None, color=None):
        self.editor = editor    # 結果出力用エディタ
        self.out = out       # 標準出力・標準エラーなどの出力オブジェクト
        # 結果出力時の色(Noneが指定されている場合、エディタの現在の色を入れる)
        if not color:
            self.color = editor.textColor()
        else:
            self.color = color

    def write(self, message):
        # カーソルを文末に移動。
        self.editor.moveCursor(QtGui.QTextCursor.End)

        # color変数に値があれば、元カラーを残してからテキストのカラーを
        # 変更する。
        self.editor.setTextColor(self.color)

        # 文末にテキストを追加。
        self.editor.insertPlainText(message)

        # 出力オブジェクトが指定されている場合、そのオブジェクトにmessageを
        # 書き出す。
        if self.out:
            self.out.write(message)

    def flush(self):
        pass


class RightDockWidget(QWidget):
    def __init__(self, parent=None):
        super(RightDockWidget, self).__init__(parent)

        self.SLMButton = QPushButton('SLM ON', self)
        self.SLMButton.setCheckable(True)
        self.SLMButton.toggled.connect(self.switch_SLM)
        self.modulateButton = QPushButton('Modulation', self)
        self.modulateButton.setCheckable(True)
        self.modulateButton.toggled.connect(self.modulate_SLM)
        self.wavelengthBox = QComboBox(self)
        self.wavelengthBox.addItems(("1053nm", "1064nm"))
        self.wavelengthBox.currentIndexChanged.connect(self.wavelengthChanged)
        self.pitchBox = QSpinBox(self)
        self.pitchBox.setMinimum(1)
        self.pitchBox.setValue(23)
        self.pitchBox.valueChanged.connect(self.pitchChanged)
        self.SLMDial = QDial(self)
        self.SLMDial.setWrapping(True)
        self.SLMDial.setMaximum(360)
        self.SLMDial.setMinimum(0)
        self.SLMDial.valueChanged.connect(self.dialChanged)
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

        hbox = QHBoxLayout()
        hbox.addWidget(self.SLMButton)
        hbox.addWidget(self.modulateButton)
        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel('Wavelength:'), alignment=Qt.AlignRight)
        hbox1.addWidget(self.wavelengthBox)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel('Pitch:'), alignment=Qt.AlignRight)
        hbox2.addWidget(self.pitchBox)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(QLabel('Focus:'), alignment=Qt.AlignRight)
        hbox3.addWidget(self.focusBox)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(QLabel('FocusPoint:'), alignment=Qt.AlignRight)
        hbox4.addWidget(self.focusXBox)
        hbox4.addWidget(self.focusYBox)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addWidget(self.SLMDial)

        self.correctImg = [r'..\..\CAL_LSH0701554_1050nm.bmp', r'..\..\CAL_LSH0701554_1060nm.bmp']
        self.alphaList = [213, 215]
        self.correction = cv2.imread(self.correctImg[0], 0)
        self.alpha = self.alphaList[0]
        self.base = self.correction
        self.theta = 0
        self.pitch = 23
        self.focus = 0
        self.focusX = 396
        self.focusY = 300
        self.w = SLMWindow(self)


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

    def dialChanged(self, val):
        self.theta = val/180*np.pi
        if self.SLMButton.isChecked():
            if self.modulateButton.isChecked():
                self.update_img()

    def pitchChanged(self, val):
        self.pitch = val
        if self.SLMButton.isChecked():
            if self.modulateButton.isChecked():
                self.update_img()

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


class BottomDockWidget(QWidget):
    def __init__(self, parent=None):
        super(BottomDockWidget, self).__init__(parent)
        vbox = QVBoxLayout(self)
        resultTE = QTextEdit(self)
        resultTE.setReadOnly(True)            # 編集不可に設定。
        resultTE.setUndoRedoEnabled(False)    # Undo・Redo不可に設定。

        # 標準出力と標準エラー出力の結果を結果出力窓に書き出すよう関連付ける。
        sys.stdout = Logger(
            resultTE, sys.stdout
        )
        sys.stderr = Logger(
            resultTE, sys.stderr, QtGui.QColor(255, 0, 0)
        )

        vbox.addWidget(resultTE)
        self.setLayout(vbox)


class mainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)

        menu = self.menuBar()
        fileMenu = menu.addMenu('File')

        self.toolbar = self.addToolBar('Toolbar')


        self.central = centralWidget(self)
        self.setCentralWidget(self.central)

        self.right_dock = QDockWidget('SLM Controler', self)
        self.right_dock.setWidget(RightDockWidget(self))
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        self.bottom_dock = QDockWidget("Log", self)
        self.bottom_dock.setWidget(BottomDockWidget(self))
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

        expAct = QAction('Export img', self)
        expAct.triggered.connect(self.central.exportBMP)
        exitAct = QAction('Exit', self)
        exitAct.triggered.connect(self.close)
        fileMenu.addAction(expAct)
        fileMenu.addAction(exitAct)

        self.setGeometry(100, 100, 1200, 800)
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
    sys.exit(app.exec_())
