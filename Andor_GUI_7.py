# -*- coding: utf-8 -*-

import ctypes as ct
import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
import logging
import socket
import mmap
import gaussfit as gf
import matplotlib.pyplot as plt
from glob import glob
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir, Qt, QSettings, QTimer
from PyQt5 import QtGui
from PyQt5 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg\
 import FigureCanvasQTAgg as FigureCanvas

# set current directory
if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

simMode = True
if socket.gethostname() == "EXPERIMENT2":
    simMode = False
    os.chdir(r'../Andor_dll/Andor_dll')
else:
    os.chdir(r'Andor_dll\Andor_dll')

# ctypes prototype declaration

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
dll.GetEnumCount.argtypes = [ct.c_int, ct.c_wchar_p, ct.POINTER(ct.c_int)]
dll.GetEnumStringByIndex.argtypes = [ct.c_int, ct.c_wchar_p, ct.c_int, ct.c_wchar_p, ct.c_int]
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

if not simMode:
    dll.setPiezoServo.argtypes = (ct.c_int, ct.c_int)
    dll.movePiezo.argtypes = (ct.c_int, ct.c_double)
    dll.getPiezoPosition.argtypes = (ct.c_int, ct.POINTER(ct.c_double))

# Class definition for multithreading

class ImageAcquirer(QtCore.QThread):

    posSignal = QtCore.pyqtSignal(float, float)
    imgSignal = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stopped = False
        self.stopDll = ct.c_bool(False)
        self.mutex = QtCore.QMutex()

    def setup(self, Handle, dir=None, fixed=0, mode=0, center=None, **args):
        self.Handle = Handle
        self.dir = dir
        self.fixed = fixed
        self.mode = mode
        self.args = args
        self.areaIsSelected = False
        self.selectedAreas = []
        if center is None:
            self.center = (ct.c_float*2)(0.0, 0.0)
        else:
            self.center = (ct.c_float*2)(center[0], center[1])
        self.stopDll.value = False
        self.stopped = False

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopDll.value = True
            self.stopped = True

    def run(self):
        if self.stopped:
            return
        if self.fixed == 0:
            if self.mode==0:
                dll.startFixedAcquisitionFile(self.Handle, self.dir, self.args["num"], self.args["count_p"], ct.byref(self.stopDll))
            elif self.mode==1:
                self.prepareAcquisition()
            elif self.mode==2:
                self.feedbackedAcquisition()
            else:
                dll.startFixedAcquisitionFilePiezo(self.Handle, self.dir, self.args["num"], self.args["count_p"], ct.byref(self.stopDll), self.args["piezoID"])
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

        ImageHeight, ImageWidth, ImageStride = ct.c_longlong(), ct.c_longlong(), ct.c_longlong()

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
        maxVal, minVal = ct.c_double(), ct.c_double()

        while not self.stopped:
            if dll.WaitBuffer(self.Handle, ct.byref(Buffer), ct.byref(BufferSize), 10000) == 0:
                ret = dll.convertBuffer(Buffer, outputBuffer, ImageWidth, ImageHeight, ImageStride)
                dll.processImageShow(ImageHeight, ImageWidth, outputBuffer, self.prms,
                                     outBuffer, ct.byref(maxVal), ct.byref(minVal))
                self.max, self.min = maxVal.value, minVal.value
                self.img = np.array(outBuffer).reshape(ImageHeight.value, ImageWidth.value)
                if self.areaIsSelected:
                    for area in self.selectedAreas:
                        self.img = cv2.rectangle(self.img, area[0], area[1], 127, 3)
                self.width, self.height = ImageWidth.value, ImageHeight.value
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
        dll.multithread(self.Handle, self.dir, self.args["num"], self.args["count_p"], self.prms,
        point, self.center, ct.c_float(self.args["DIOthres"]), self.args["DIOhandle"], ct.c_int(self.mode))
        self.posSignal.emit(point[0], point[1])

    def feedbackedAcquisition(self):
        point = (ct.c_float*2)()
        dll.multithread(self.Handle, self.dir, self.args["num"], self.args["count_p"], self.prms,
        point, self.center, ct.c_float(self.args["DIOthres"]), self.args["DIOhandle"], ct.c_int(self.mode))


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


class ImageProcessor(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stopped = False
        self.mutex = QtCore.QMutex()

    def setup(self, datfiles, width, height, stride, prms, gfchecked, selectedAreas, dir, pBar, old, mm=None):
        self.datfiles = datfiles
        self.width = width
        self.height = height
        self.stride = stride
        self.prms = prms
        self.gfchecked = gfchecked
        self.selectedAreas = selectedAreas
        self.dir = dir
        self.pBar = pBar
        self.old = old
        self.mm = mm
        self.stopped = False

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        if self.stopped:
            return
        if self.old:
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
                ## z-position detection

                self.pBar.setValue(i)
            DF = pd.DataFrame(np.array(pointlst))
            DF.columns = ['x', 'y']
            DF.to_csv(self.dir + r"\COG.csv")
            logging.info("analysis finished")
        else:
            ImgArry = (ct.c_ushort * (self.width * self.height))()
            point = (ct.c_float*2)()
            pointlst = []
            prmsList = []
            datfile, start, end, imgSize = self.datfiles
            num = end - start
            self.pBar.setMaximum(num-1)
            self.mm.seek(imgSize*start)
            logText = ""
            for i in range(num):
                rawdata = self.mm.read(imgSize)
                buffer = ct.cast(rawdata, ct.POINTER(ct.c_ubyte))
                ret = dll.convertBuffer(buffer, ImgArry, self.width, self.height, self.stride)
                if self.gfchecked:
                    imgNpArray = np.array(ImgArry).reshape(self.width, self.height)
                    for area, j in zip(self.selectedAreas, range(len(self.selectedAreas))):
                        prmList = []
                        LT, RB = area
                        width = RB[0] - LT[0]
                        height = RB[1] - LT[1]
                        areaImg = imgNpArray[LT[1]:RB[1], LT[0]:RB[0]]
                        gaussfit = gf.GaussFit(areaImg)
                        try:
                            fitted, prms, cov = gaussfit.fit()
                        except RuntimeError:
                            prmList.append(i)
                            prmList.append(j)
                            prmList += [None]*14
                            prmsList.append(prmList)
                            logText += f"Fitting failed at area {j} in shot {i}\n"
                        else:
                            prmList.append(i)
                            prmList.append(j)
                            prmList.append(LT[0])
                            prmList.append(LT[1])
                            prmList += list(prms)
                            err = np.sqrt(np.diag(cov))
                            prmList += list(err)
                            prmsList.append(prmList)
                    self.pBar.setValue(i)
                else:
                    if self.selectedAreas:
                        imgNpArray = np.array(ImgArry).reshape(self.width, self.height)
                        for area, j in zip(self.selectedAreas, range(len(self.selectedAreas))):
                            prmList = []
                            LT, RB = area
                            width = RB[0] - LT[0]
                            height = RB[1] - LT[1]
                            areaImg = imgNpArray[LT[1]:RB[1], LT[0]:RB[0]]
                            areaImg_np = np.copy(areaImg)
                            areaImgC = areaImg_np.ctypes.data_as(ct.POINTER(ct.c_ushort * (width*height)))
                            areaImgC_pt = ct.cast(areaImgC, ct.POINTER(ct.c_ushort))
                            point = (ct.c_float*2)()
                            dll.processImage(point, height, width, areaImgC_pt, self.prms)
                            pointlst.append([point[0]+LT[0], point[1]+LT[1]])
                        self.pBar.setValue(i)

                    else:
                        dll.processImage(point, self.height, self.width, ImgArry, self.prms)
                        pointlst.append([point[0], point[1]])
                        self.pBar.setValue(i)
            if pointlst:
                DF = pd.DataFrame(np.array(pointlst).reshape(num, -1))
                columnNames = ["x/Area"+str(i//2) if i%2==0 else "y/Area"+str(i//2) for i in range(DF.shape[1])]
                DF.columns = columnNames
                DF.to_csv(self.dir + r"\COG.csv", index=False)
            if prmsList:
                DF = pd.DataFrame(np.array(prmsList))
                columnNames = ["shotNumber", "areaNumber", "Left", "Top", "Amp.", "Cx", "Cy", "Sx", "Sy", "Base", "E_Amp.", "E_Cx", "E_Cy", "E_Sx", "E_Sy", "E_Base"]
                DF.columns = columnNames
                DF.to_csv(self.dir + r"\FittingResults.csv", index=False)
                with open(self.dir + r"\FittingLog.txt", "w") as f:
                    f.write(logText)
            logging.info("analysis finished")
        self.stopped = True


# class definition for UI

def setListToLayout(layout, list):
    for object in list:
        if isinstance(object, QLayout):
            layout.addLayout(object)
        else:
            layout.addWidget(object)


# Layout of labeled widget
class LHLayout(QHBoxLayout):
    def __init__(self, label, object, parent=None):
        super(QHBoxLayout, self).__init__(parent)
        self.addWidget(QLabel(label), alignment=Qt.AlignRight)
        if isinstance(object, list):
            setListToLayout(self, object)
        elif isinstance(object, QLayout):
            self.addLayout(object)
        else:
            self.addWidget(object)


class MyImageWidget(QWidget):
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

        self.imageWidget = MyImageWidget(self)
        self.posLabel = QLabel("Position:")

        vbox = QVBoxLayout(self)
        setListToLayout(vbox, [self.imageWidget, self.posLabel])
        vbox.addStretch(1)

    def writeLabel(self, string):
        self.posLabel.setText(string)
        self.update()

    def setImage(self, image):
        self.imageWidget.setImage(image)


class SLMWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MyImageWidget(self)
        self.canvas.setMinimumSize(792,600)

        self.img = None

        desktop = app.desktop()
        left = desktop.availableGeometry(1).left()
        top = desktop.availableGeometry(1).top()
        self.move(left, top)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.BypassWindowManagerHint)

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
        self.imgMinBox = QLineEdit(self)
        self.imgMaxBox.setReadOnly(True)
        self.imgMinBox.setReadOnly(True)
        self.imgMaxBox.setSizePolicy((QSizePolicy(5, 0)))
        self.imgMinBox.setSizePolicy((QSizePolicy(5, 0)))

        self.markerPositionBoxX = QSpinBox(self)
        self.markerPositionBoxY = QSpinBox(self)
        self.markerPositionBoxX.setMaximum(396)
        self.markerPositionBoxY.setMaximum(396)
        self.markerPositionBoxX.setMinimum(-396)
        self.markerPositionBoxY.setMinimum(-396)
        self.markerFactorBox = QSpinBox(self)
        self.markerFactorBox.setRange(0, 9999)
        self.splitButton = QCheckBox("Split", self)

        self.initLayout()

    def initLayout(self):

        RawImgLayout = QVBoxLayout()
        RawImgGroup = QGroupBox("Raw Image", self)
        RawImgGroup.setLayout(RawImgLayout)
        setListToLayout(RawImgLayout, [LHLayout('Min: ', self.imgMinBox), LHLayout('Max: ', self.imgMaxBox)])

        MarkerLayout = QVBoxLayout()
        posLayout = QGridLayout()
        posLayout.addWidget(QLabel("Position"), 0, 0)
        posLayout.addLayout(LHLayout('x: ', self.markerPositionBoxX), 0, 1)
        posLayout.addLayout(LHLayout('y: ', self.markerPositionBoxY), 1, 1)
        splitLayout = QHBoxLayout()
        splitLayout.addWidget(self.splitButton)
        splitLayout.addLayout(LHLayout('factor: ', self.markerFactorBox))
        MarkerGroup = QGroupBox("Marker", self)
        MarkerGroup.setLayout(MarkerLayout)
        setListToLayout(MarkerLayout, [posLayout, splitLayout])

        MainLayout = QHBoxLayout(self)
        setListToLayout(MainLayout, [RawImgGroup, MarkerGroup])
        MainLayout.setStretch(0, 1)
        MainLayout.setStretch(1, 1)


class CommentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.textEdit = QTextEdit(self)
        self.applyButton = QPushButton("Apply")
        self.cancelButton = QPushButton("Cancel")

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.applyButton)
        bottomLayout.addWidget(self.cancelButton)
        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(self.textEdit)
        mainLayout.addLayout(bottomLayout)

        self.applyButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)


class fixedWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.comment = ""

        self.initUI()

    def initUI(self):
        self.numImgBox = QLineEdit('100', self)
        self.frameRateBox = QLineEdit('30', self)
        self.countBox = QLineEdit(self)
        self.countBox.setReadOnly(True)
        self.dirBox = QLineEdit(self)
        self.dirBox.setReadOnly(True)
        self.dirButton = QPushButton('...', self)
        self.dirButton.setMaximumWidth(20)
        self.dirButton.clicked.connect(self.selectDirectory)
        self.aqTypeBox = QComboBox(self)
        aqTypeList = ['No feed-back', 'Preparation', 'Do feed-back']
        self.aqTypeBox.addItems(aqTypeList)
        self.cntrPosBox = QLineEdit(self)
        self.thresBox = QDoubleSpinBox(self)
        self.specialButton = QPushButton('Special Measurement', self)
        self.specialButton.setCheckable(True)
        self.currentRepeatBox = QLineEdit(self)
        self.currentRepeatBox.setReadOnly(True)
        self.currentRepeatBox.setSizePolicy(QSizePolicy(5, 0))
        self.commentButton = QPushButton("Comment")
        self.commentButton.setCheckable(True)
        self.commentButton.toggled.connect(self.commentEdit)
        self.commentDialog = CommentDialog(self)
        self.commentDialog.setWindowTitle("Comment")

        self.initLayout()

    def initLayout(self):
        leftGridLayout = QGridLayout()
        leftGridLayout.addWidget(QLabel("Frames:"), 0, 0)
        leftGridLayout.addWidget(self.numImgBox, 0, 1)
        leftGridLayout.addWidget(QLabel("Frame Rate:"), 1, 0)
        leftGridLayout.addWidget(self.frameRateBox, 1, 1)
        dirLayout = QHBoxLayout()
        dirLayout.addWidget(self.dirBox)
        dirLayout.addWidget(self.dirButton)
        leftGridLayout.addWidget(QLabel("Directory:"), 2, 0)
        leftGridLayout.addLayout(dirLayout, 2, 1)
        leftGridLayout.addWidget(QLabel("Current Frame:"), 3, 0)
        leftGridLayout.addWidget(self.countBox, 3, 1)
        leftGridLayout.addWidget(self.commentButton, 4, 0)
        leftGridLayout.setColumnStretch(0, 1)
        leftGridLayout.setColumnStretch(1, 1)

        rightGridLayout = QGridLayout()
        i = 0
        for label, widget in [("Feedback:", self.aqTypeBox), ("Potential Center:", self.cntrPosBox), ("Threshold: ", self.thresBox)]:
            rightGridLayout.addWidget(QLabel(label), i, 0)
            rightGridLayout.addWidget(widget, i, 1)
            i += 1
        rightGridLayout.addWidget(self.specialButton, 3, 0)
        rightGridLayout.addWidget(self.currentRepeatBox, 3, 1)
        rightGridLayout.setColumnStretch(0, 1)
        rightGridLayout.setColumnStretch(1, 1)

        mainLayout = QHBoxLayout(self)
        mainLayout.addLayout(leftGridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(rightGridLayout)

    def selectDirectory(self):
        self.dirname = QFileDialog.getExistingDirectory(self, 'Select directory', self.dirname)
        self.dirname = QDir.toNativeSeparators(self.dirname)
        self.dirBox.setText(self.dirname)

    def commentEdit(self, checked):
        if checked:
            if self.commentDialog.exec_():
                self.comment = self.commentDialog.textEdit.document()
            else:
                self.commentButton.setChecked(False)


class AOISettingBox(QGroupBox):

    def __init__(self, parent=None):
        super().__init__("AOI Settings", parent)

        self.defaultCheck = QRadioButton("Default")
        self.defaultCheck.setChecked(True)

        self.AOISizeBox = QComboBox(self)
        AOISizeList = ['2048x2048','1024x1024','512x512','256x256','128x128']
        self.AOISizeBox.addItems(AOISizeList)

        self.CenterXBox = QSpinBox(self)
        self.CenterYBox = QSpinBox(self)
        self.CenterXBox.setMinimum(-1024)
        self.CenterYBox.setMinimum(-1024)
        self.CenterXBox.setMaximum(1024)
        self.CenterYBox.setMaximum(1024)

        self.customCheck = QRadioButton("Customized")

        self.AOILeftBox = QSpinBox(self)
        self.AOITopBox = QSpinBox(self)
        self.AOIWidthBox = QSpinBox(self)
        self.AOIHeightBox = QSpinBox(self)
        AOIBoxList = [self.AOITopBox, self.AOILeftBox, self.AOIWidthBox, self.AOIHeightBox]
        for obj in AOIBoxList:
            obj.setMaximum(2048)

        self.AOIButtons = QButtonGroup(self)
        self.AOIButtons.addButton(self.defaultCheck)
        self.AOIButtons.addButton(self.customCheck)
        self.AOIButtons.setExclusive(True)
        self.AOIButtons.setId(self.defaultCheck, 0)
        self.AOIButtons.setId(self.customCheck, 1)

        defaultLayout = QVBoxLayout()
        defaultLayout.addWidget(self.defaultCheck)
        defaultLayout.addLayout(LHLayout("Size:", self.AOISizeBox))
        defaultCenterLayout = QHBoxLayout()
        setListToLayout(defaultCenterLayout, [LHLayout("X Center:", self.CenterXBox), LHLayout("Y Center:", self.CenterYBox)])
        defaultLayout.addLayout(defaultCenterLayout)

        customLayout = QVBoxLayout()
        customLayout.addWidget(self.customCheck)
        customParamsLayout = QGridLayout()
        customParamsLayout.addLayout(LHLayout("Left: ", self.AOILeftBox), 0, 1)
        customParamsLayout.addLayout(LHLayout("Top: ", self.AOITopBox), 0, 0)
        customParamsLayout.addLayout(LHLayout("Width: ", self.AOIWidthBox), 1, 0)
        customParamsLayout.addLayout(LHLayout("Height: ", self.AOIHeightBox), 1, 1)
        customLayout.addLayout(customParamsLayout)

        MainLayout = QHBoxLayout(self)
        MainLayout.addLayout(defaultLayout)
        MainLayout.addSpacing(20)
        MainLayout.addLayout(customLayout)

class AcquisitionWidget(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.contWidget = contWidget(self)
        self.fixedWidget = fixedWidget(self)

        self.AOI = AOISettingBox(self)
        self.AOIWidth = self.AOIHeight = 2048

        self.Tab = QTabWidget()
        self.Tab.addTab(self.contWidget, 'Live Image')
        self.Tab.addTab(self.fixedWidget, 'Recording')

        self.initUI()
        # self.AOI.AOIButtons.buttonClicked.connect(self.setAOISize)

    def initUI(self):
        self.handleBox = QLineEdit(self)
        self.handleBox.setReadOnly(True)
        self.handleBox.setSizePolicy(QSizePolicy(5, 0))
        self.exposeTBox = QDoubleSpinBox(self)
        self.exposeTBox.setDecimals(5)

        self.AOIBinBox = QComboBox(self)
        AOIBinList = ["1x1", "2x2", "3x3", "4x4", "8x8"]
        self.AOIBinBox.addItems(AOIBinList)
        self.globalClearButton = QPushButton('Global clear', self)
        self.globalClearButton.setCheckable(True)

        self.initButton = QPushButton("CONNECT", self)
        self.applyButton = QPushButton('APPLY', self)
        self.applyButton.setEnabled(False)
        self.runButton = QPushButton('RUN', self)
        self.runButton.setCheckable(True)
        self.runButton.setEnabled(False)
        self.finButton = QPushButton("DISCONNECT", self)
        self.finButton.setEnabled(False)

        self.initLayout()

    def initLayout(self):
        topLayout = QHBoxLayout()
        setListToLayout(topLayout, [LHLayout("Handle: ", self.handleBox), LHLayout('Exposure Time (s): ', self.exposeTBox),\
                                 LHLayout("Binnng: ", self.AOIBinBox), self.globalClearButton])

        bottomLayout = QHBoxLayout()
        setListToLayout(bottomLayout, [self.initButton, self.applyButton, self.runButton, self.finButton])

        mainLayout = QVBoxLayout(self)
        setListToLayout(mainLayout, [topLayout, self.AOI, self.Tab, bottomLayout])

    # def setAOISize(self, button):
    #     if button is self.AOI.customCheck:
    #         self.AOIWidth = self.AOI.AOIWidthBox.value()
    #         self.AOIHeight = self.AOI.AOIHeightBox.value()
    #         logging.debug("id=1")
    #     else:
    #         index = self.AOI.AOISizeBox.currentIndex()
    #         val = 2048/(2**index)
    #         logging.debug("id=0")
    #         self.AOIWidth = val
    #         self.AOIHeight = val


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

        self.gaussFitButton = QPushButton("Gauss Fit")
        self.gaussFitButton.setCheckable(True)

        self.anlzButton = QPushButton('Analyse Position', self)
        self.anlzStartBox = QSpinBox(self)
        self.anlzEndBox = QSpinBox(self)
        self.anlzCheck = QCheckBox('Multiple Directories', self)
        self.anlzFilter = QLineEdit(self)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)

        self.initLayout()

    def initLayout(self):
        centerLayout = QGridLayout()
        centerLayout.addLayout(LHLayout("Frames: ", self.fileNumBox), 0, 0)
        centerLayout.addLayout(LHLayout("Current Frame: ", self.currentNumBox), 0, 1)
        centerLayout.addLayout(LHLayout("Min: ", self.imgMinBox), 1, 0)
        centerLayout.addLayout(LHLayout("Max: ", self.imgMaxBox), 1, 1)
        centerLayout.addLayout(LHLayout("Start: ", self.anlzStartBox), 2, 0)
        centerLayout.addLayout(LHLayout("End: ", self.anlzEndBox), 2, 1)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.anlzButton)
        bottomLayout.addWidget(self.anlzCheck)
        bottomLayout.addWidget(self.anlzFilter)

        vbox0 = QVBoxLayout(self)
        setListToLayout(vbox0, [LHLayout("Directory: ", [self.dirBox, self.dirButton]), centerLayout, self.gaussFitButton, bottomLayout, self.progressBar])

    def initVal(self):
        self.dirname = None
        self.img = None
        self.mm = None
        self.f = None
        self.old = True

    def selectDirectory(self):
        dirname = QFileDialog.getExistingDirectory(self, 'Select directory', self.dirname)
        dirname = QDir.toNativeSeparators(dirname)
        self.dirBox.setText(dirname)
        metafile = dirname + r'\metaSpool.txt'
        if os.path.isfile(metafile):
            if not self.mm is None:
                self.mm.close()
                self.f.close()
            self.dirname = dirname
            f = open(metafile, mode='r')
            metadata = f.readlines()
            f.close()
            if len(metadata) != 7:
                self.old = True
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
                self.old = False
                self.datfiles = glob(dirname + r"\*.dat")
                self.filenum = len(self.datfiles)
                self.imgSize = int(metadata[0])
                self.encoding = metadata[1]
                self.stride = int(metadata[2])
                self.height = int(metadata[3])
                self.width = int(metadata[4])
                self.frameNum = int(metadata[6])
                self.fileNumBox.setText(str(self.frameNum))
                self.anlzStartBox.setMaximum(self.frameNum-1)
                self.anlzEndBox.setMaximum(self.frameNum-1)
                self.anlzEndBox.setValue(self.frameNum-1)
                self.currentNumBox.setMaximum(self.frameNum - 1)
                self.f = open(dirname + "/spool.dat", mode="r+b")
                self.mm = mmap.mmap(self.f.fileno(), 0)
                self.update_img()
        else:
            logging.error('No metadata was found.\n')

    def update_frame(self):
        img = self.img
        img_height, img_width = img.shape
        scale_w = float(600) / float(img_width)
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
        if self.old:
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
        else:
            num = int(self.currentNumBox.text())
            self.mm.seek(self.imgSize*num)
            rawdata = self.mm.read(self.imgSize)
            buffer = ct.cast(rawdata, ct.POINTER(ct.c_ubyte))
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
            if self.areaIsSelected:
                for area in self.selectedAreas:
                    self.img = cv2.rectangle(self.img, area[0], area[1], 127, 3)
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
        self.areaSelectButton = QPushButton("Area Select")
        self.areaSelectButton.setCheckable(True)
        self.contNum = QSpinBox(self)
        self.contNum.setRange(1,4)

        self.applyButton = QPushButton('APPLY')

        self.initLayout()

    def initLayout(self):
        # gbox = QGridLayout()
        # gbox.addWidget(self.normButton, 0, 0)
        # gbox.addLayout(LHLayout("Min: ", self.normMin), 0, 1)
        # gbox.addLayout(LHLayout("Max: ", self.normMax), 0, 2)
        #
        # gbox.addWidget(self.standButton, 0, 3)
        # gbox.addLayout(LHLayout("Sigma: ", self.sigma), 0, 5)
        #
        # gbox.addWidget(self.blurButton, 1, 0)
        # gbox.addLayout(LHLayout("Kernel size: ", self.blurSize), 1, 3)
        # gbox.addLayout(LHLayout("Sigma: ", self.blurSigma), 1, 5)
        #
        # gbox.addWidget(self.thresButton, 2, 0)
        # gbox.addLayout(LHLayout("Threshold: ", self.thresVal), 2, 2)
        # gbox.addLayout(LHLayout("Type: ", self.thresType), 2, 4)
        # gbox.addWidget(self.OtsuButton, 2, 5)
        #
        # gbox.addWidget(self.contButton, 3, 0)
        # gbox.addLayout(LHLayout("Number to Find: ", self.contNum), 3, 3)

        hbox000 = QHBoxLayout()
        setListToLayout(hbox000, [self.normButton, LHLayout("Min: ", self.normMin),\
                                  LHLayout("Max: ", self.normMax)])

        hbox001 = QHBoxLayout()
        hbox001.addWidget(self.standButton)
        hbox001.addLayout(LHLayout("Sigma: ", self.sigma))

        hbox00 = QHBoxLayout()
        hbox00.addLayout(hbox000)
        hbox00.addLayout(hbox001)

        hbox01 = QHBoxLayout()
        setListToLayout(hbox01, [self.blurButton, LHLayout("Kernel size : ", self.blurSize),\
                                 LHLayout("Sigma: ", self.blurSigma)])

        hbox02 = QHBoxLayout()
        setListToLayout(hbox02, [self.thresButton, LHLayout("Threshold: ", self.thresVal),\
                        LHLayout("Type: ", self.thresType), self.OtsuButton])

        hbox03 = QHBoxLayout()
        hbox03.addWidget(self.contButton)
        hbox03.addWidget(self.areaSelectButton)
        hbox03.addLayout(LHLayout("Number to Find: ", self.contNum))

        vbox0 = QVBoxLayout(self)
        # vbox0.addLayout(gbox)
        # vbox0.addWidget(self.applyButton)
        setListToLayout(vbox0, [hbox00, hbox01, hbox02, hbox03, self.applyButton])

        # self.setStyleSheet("background-color:white;")
        self.setTitle("Process Settings")

    def initVal(self):
        self.prmStruct = processPrms(nsPrms(0,0,0,0),
        blurPrms(0,0,0,0),
        thresPrms(0,0,0,0),
        contPrms(0,0))
        self.areaIsSelected = False
        self.selectedAreas = []

    def set_prm(self):
        self.areaIsSelected = self.areaSelectButton.isChecked()
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

        self.pitchBox = QDoubleSpinBox(self)
        self.pitchBox.setMinimum(1)

        self.rotationBox = QSpinBox(self)
        self.rotationBox.setMaximum(359)

        self.SLMDial = QDial(self)
        self.SLMDial.setWrapping(True)
        self.SLMDial.setMaximum(360)
        self.SLMDial.setMinimum(0)
        self.SLMDial.setMinimumSize(150, 150)

        self.rotationBox.valueChanged[int].connect(self.SLMDial.setValue)
        self.SLMDial.valueChanged[int].connect(self.rotationBox.setValue)

        self.tiltXBox = QDoubleSpinBox(self)
        self.tiltYBox = QDoubleSpinBox(self)

        self.tiltXBox.valueChanged.connect(self.tiltXChanged)
        self.tiltYBox.valueChanged.connect(self.tiltYChanged)

        for SpinBox in [self.tiltXBox, self.tiltYBox]:
            SpinBox.setMinimum(-20.0)
            SpinBox.setSingleStep(0.1)

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

        self.alignmentToolButton = QPushButton("Alignment Tool", self)
        self.alignmentToolButton.setCheckable(True)
        self.alignmentToolButton.toggled.connect(self.switchMask)
        self.aspectBox = QDoubleSpinBox(self)
        self.aspectBox.setDecimals(3)
        self.aspectBox.setSingleStep(0.01)
        self.aspectBox.valueChanged.connect(self.aspectChanged)

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

        RotationGroup = QGroupBox("Rotation", self)
        RotationLayout = QVBoxLayout()
        RotationGroup.setLayout(RotationLayout)
        RotationLayout.addWidget(self.SLMDial)
        RotationLayout.addWidget(self.rotationBox)
        vbox00 = QVBoxLayout()
        vbox00.addWidget(RotationGroup)

        TiltBox = QGroupBox("Tilt", self)
        TiltLayout = QHBoxLayout()
        TiltBox.setLayout(TiltLayout)
        TiltLayout.addLayout(LHLayout("x:", self.tiltXBox))
        TiltLayout.addLayout(LHLayout("y:", self.tiltYBox))
        vbox00.addWidget(TiltBox)

        vbox01 = QVBoxLayout()
        setListToLayout(vbox01, [hbox010, hbox011, hbox012, hbox013, hbox014, LHLayout("Aspect Ratio", self.aspectBox), self.alignmentToolButton])

        hbox0 = QHBoxLayout(self)
        hbox0.addLayout(vbox01)
        hbox0.addSpacing(20)
        hbox0.addLayout(vbox00)

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
        self.tiltX = 0
        self.tiltY = 0
        self.mask = 0
        self.aspectRatio = 1.0

    def switch_SLM(self, checked):
        if checked:
            self.focus = self.focusBox.value()
            self.w.showFullScreen()
            self.make_base()
            self.init_img()
            self.w.update_SLM()
            logging.debug((self.w.geometry()))
        else:
            self.w.hide()

    def wavelengthChanged(self, index):
        self.correction = cv2.imread(self.correctImg[index], 0)
        self.alpha = self.alphaList[index]

    def make_base(self):
        x = (np.arange(792)-self.focusX)*self.aspectRatio
        y = np.arange(600)-self.focusY
        X, Y = np.meshgrid(x, y)
        focusCorrection = (X**2 + Y**2)*self.focus*1e-2
        tiltCorrection = self.tiltX*X + self.tiltY*Y
        self.base = (self.correction + tiltCorrection + focusCorrection + self.mask).astype(np.uint8)

    def init_img(self):
        img = self.base.astype(np.float) * self.alpha / 255
        self.w.img = img.astype(np.uint8)

    def update_img(self):
        x = np.arange(792)*self.aspectRatio
        y = np.arange(600)
        X, Y = np.meshgrid(x, y)

        rotX = X*np.sin(self.theta) + Y*np.cos(self.theta)
        rotY = X*np.cos(self.theta) - Y*np.sin(self.theta)

        img = ((rotX // self.pitch + rotY // self.pitch) % 2) * 128
        img = (img + self.base).astype(np.uint8)
        img = img.astype(np.float) * self.alpha / 255
        self.w.img = img.astype(np.uint8)
        self.w.update_SLM()

    def modulate_SLM(self, checked):
        if checked:
            self.update_img()
        else:
            self.init_img()
            self.w.update_SLM()

    def tiltXChanged(self, val):
        self.tiltX = val
        self.make_base()
        if self.SLMButton.isChecked():
            self.modulate_SLM(self.modulateButton.isChecked())

    def tiltYChanged(self, val):
        self.tiltY = val
        self.make_base()
        if self.SLMButton.isChecked():
            self.modulate_SLM(self.modulateButton.isChecked())

    def focusChanged(self, val):
        self.focus = self.focusBox.value()
        self.focusX = self.focusXBox.value()+396
        self.focusY = self.focusYBox.value()+300
        self.make_base()
        if self.SLMButton.isChecked():
            self.modulate_SLM(self.modulateButton.isChecked())

    def switchMask(self, checked):
        if checked:
            x = np.arange(792)
            y = np.arange(600)
            X, Y = np.meshgrid(x, y)
            pitch = 1

            img = ((X // pitch + Y // pitch) % 2) * 128
            circle = (X-396)**2 + (Y-300)**2
            thres = circle > 200**2
            self.mask = thres*img.astype(np.uint8)
        else:
            self.mask = 0
        self.make_base()
        if self.SLMButton.isChecked():
            self.modulate_SLM(self.modulateButton.isChecked())

    def aspectChanged(self, val):
        self.aspectRatio = self.aspectBox.value()
        self.make_base()
        if self.SLMButton.isChecked():
            self.modulate_SLM(self.modulateButton.isChecked())


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
        vbox.addWidget(self.DIOHighButton)
        vbox.addWidget(self.DIOLowButton)

        self.DIOLowButton.setChecked(True)
        self.setTitle("EOM Controller")


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
        vbox.addWidget(self.openButton)
        vbox.addWidget(self.closeButton)

        self.closeButton.setChecked(True)
        self.setTitle("Shutter Controller")

    def change(self):
        if self.openButton.isChecked():
            self.closeButton.click()
        else:
            self.openButton.click()
        QApplication.processEvents()


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

class TemperatureWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tempButton = QPushButton('Show Tempereture', self)
        self.tempStatusButton = QPushButton('Show Status', self)
        self.sensorCoolingButton = QPushButton('Sensor Cooling', self)
        self.sensorCoolingButton.setCheckable(True)

        hbox = QHBoxLayout(self)
        setListToLayout(hbox, [self.tempButton, self.tempStatusButton, self.sensorCoolingButton])

        self.setTitle("Temperature Control")

class PiezoWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.connectButton = QPushButton('Connect', self)
        self.servoButton = QPushButton('Servo', self)
        self.moveButton = QPushButton('Move to', self)
        self.targetBox = QDoubleSpinBox(self)
        self.getPosButton = QPushButton('Get Position', self)
        self.piezoButtons = [self.servoButton, self.moveButton, self.targetBox, self.getPosButton]

        self.connectButton.setCheckable(True)
        self.servoButton.setCheckable(True)

        self.connectButton.toggled.connect(self.connectPiezo)
        self.servoButton.toggled.connect(self.setServo)
        self.moveButton.clicked.connect(self.movePiezo)
        self.getPosButton.clicked.connect(self.showCurrentPosition)

        for button in self.piezoButtons:
            button.setDisabled(True)

        hbox = QHBoxLayout(self)
        setListToLayout(hbox, [self.connectButton, self.servoButton, self.moveButton, self.targetBox, self.getPosButton])

        self.setTitle("Piezo Controller")

    def connectPiezo(self, checked):
        if checked:
            self.piezoID = dll.initPiezo()
            if self.piezoID<0:
                logging.error("Piezo connection failed")
                self.connectPiezo.setChecked(False)
            else:
                for button in self.piezoButtons:
                    button.setDisabled(False)
        else:
            dll.finPiezo(self.piezoID)
            for button in self.piezoButtons:
                button.setDisabled(True)

    def setServo(self, checked):
        dll.setPiezoServo(self.piezoID, checked)

    def movePiezo(self):
        target = self.targetBox.value()
        dll.movePiezo(self.piezoID, target)

    def showCurrentPosition(self):
        pos = ct.c_double()
        dll.getPiezoPosition(self.piezoID, ct.byref(pos))
        logging.info("Current Piezo Position (um):" + str(pos.value))

class SpecialMeasurementDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.mode = 0
        self.start = 0
        self.end = 0
        self.step = 0
        self.num = 0
        self.tiltXY = 0

        self.repeat = 0
        self.repeatCheck = False

        self.piezoCheckBox = QRadioButton("Piezo Measurement", self)
        self.startBox = QDoubleSpinBox(self)
        self.endBox = QDoubleSpinBox(self)
        self.stepBox = QDoubleSpinBox(self)
        self.stepBox.setDecimals(3)
        self.numBox = QSpinBox(self)

        self.tiltCheckBox = QRadioButton("Tilt Measurement", self)
        self.tiltStartBox = QDoubleSpinBox(self)
        self.tiltEndBox = QDoubleSpinBox(self)
        self.tiltStepBox = QDoubleSpinBox(self)
        self.tiltStepBox.setDecimals(3)
        self.tiltNumBox = QSpinBox(self)
        self.tiltXYBox = QComboBox(self)
        self.tiltXYBox.addItems(["X", "Y"])

        self.repeatCheckBox = QRadioButton("Cycle Measurement", self)
        self.repeatBox = QSpinBox(self)
        self.repeatBox.setMinimum(1)

        self.acceptButton = QPushButton("OK", self)
        self.rejectButton = QPushButton("Cancel", self)

        self.acceptButton.clicked.connect(self.accept)
        self.rejectButton.clicked.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addWidget(self.acceptButton)
        hbox.addWidget(self.rejectButton)

        hbox2 = QHBoxLayout()
        setListToLayout(hbox2, [self.piezoCheckBox, LHLayout("Start (um):", self.startBox),\
                                LHLayout("End (um):", self.endBox), LHLayout("Step (um):", self.stepBox),\
                                LHLayout("Number of each condition:", self.numBox)])

        hbox4 = QHBoxLayout()
        setListToLayout(hbox4, [self.tiltCheckBox, LHLayout("Start :", self.tiltStartBox),\
                                LHLayout("End :", self.tiltEndBox), LHLayout("Step :", self.tiltStepBox),\
                                LHLayout("Number of each condition:", self.tiltNumBox), self.tiltXYBox])

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.repeatCheckBox)
        hbox3.addLayout(LHLayout("Cycle:", self.repeatBox))

        vbox = QVBoxLayout(self)
        setListToLayout(vbox, [hbox2, hbox4, hbox3, hbox])

    def applySettings(self):
        if self.piezoCheckBox.isChecked():
            self.mode = 1
            self.start = self.startBox.value()
            self.end = self.endBox.value()
            self.step = self.stepBox.value()
            self.num = self.numBox.value()
        elif self.tiltCheckBox.isChecked():
            self.mode = 2
            self.start = self.tiltStartBox.value()
            self.end = self.tiltEndBox.value()
            self.step = self.tiltStepBox.value()
            self.num = self.tiltNumBox.value()
            self.tiltXY = self.tiltXYBox.currentIndex()
        else:
            self.mode=0
        self.repeatCheck = self.repeatCheckBox.isChecked()
        self.repeat = self.repeatBox.value()
        return {"mode": self.mode, "start": self.start, "end": self.end,\
                "step": self.step, "num": self.num, "tiltXY": self.tiltXY,\
                "repeatCheck": self.repeatCheck, "repeat": self.repeat}


class AreaSelectImgWidet(MyImageWidget):
    pressSignal = QtCore.pyqtSignal(QtCore.QPoint)
    releasedSignal = QtCore.pyqtSignal(QtCore.QPoint)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(False)

    def mousePressEvent(self, event):
        self.pressSignal.emit(event.pos())

    def mouseReleaseEvent(self, event):
        self.releasedSignal.emit(event.pos())


class AreaSelectDialog(QDialog):
    def __init__(self, img, parent=None):
        super().__init__(parent)

        self.imgWidget = AreaSelectImgWidet(self)
        self.imgHeight = img.shape[0]
        self.imgWidth = img.shape[1]
        self.start = None
        self.end = None
        self.rects= []

        self.imgWidget.pressSignal.connect(self.mousePressed)
        self.imgWidget.posSignal.connect(self.mouseMoved)
        self.imgWidget.releasedSignal.connect(self.mouseReleased)

        self.applyButton = QPushButton("Apply")
        self.clearButton = QPushButton("Clear")
        self.cancelButton = QPushButton("Cancel")

        self.applyButton.clicked.connect(self.accept)
        self.clearButton.clicked.connect(self.clearSelection)
        self.cancelButton.clicked.connect(self.reject)

        self.setInitImage(img)
        self.initLayout()

    def initLayout(self):
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.applyButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.cancelButton)
        layout = QVBoxLayout(self)
        layout.addWidget(self.imgWidget)
        layout.addLayout(buttonLayout)

    def setImage(self, img):
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.imgWidget.setImage(image)

    def setInitImage(self, img):
        scale_w = float(600) / float(self.imgWidth)
        scale_h = float(600) / float(self.imgHeight)
        scale = min([scale_w, scale_h])
        if scale == 0:
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        height, width, bpc = img.shape
        self.height, self.width, self.img = height, width, img
        self.setImage(img)

    def mousePressed(self, point):
        self.start = (point.x(), point.y())

    def mouseMoved(self, point):
        self.end = (point.x(), point.y())
        self.drawRectangles()

    def mouseReleased(self, point):
        self.end = (point.x(), point.y())
        length = max(abs(self.start[0]-self.end[0]), abs(self.start[1]-self.end[1]))
        if length > 0:
            LT = (self.start[0]-length, self.start[1]-length)
            RB = (self.start[0]+length, self.start[1]+length)
            self.rects.append((LT, RB))
        self.start, self.end = None, None
        self.drawRectangles()

    def drawRectangles(self):
        img = self.img.copy()
        for rect in self.rects:
            img = cv2.rectangle(img, rect[0], rect[1], (0,255,0), 3)
        if self.start and self.end:
            length = max(abs(self.start[0]-self.end[0]), abs(self.start[1]-self.end[1]))
            LT = (self.start[0]-length, self.start[1]-length)
            RB = (self.start[0]+length, self.start[1]+length)
            img = cv2.rectangle(img, LT, RB, (0,255,0), 3)
        self.setImage(img)

    def applyAreas(self):
        ret = []
        for rect in self.rects:
            LTx, LTy = min(rect[0][0], rect[1][0]), min(rect[0][1], rect[1][1])
            RBx, RBy = max(rect[0][0], rect[1][0]), max(rect[0][1], rect[1][1])
            LT = (int(LTx*self.imgWidth/self.width), int(LTy*self.imgHeight/self.height))
            RB = (int(RBx*self.imgWidth/self.width), int(RBy*self.imgHeight/self.height))
            RB = (RB[0], LT[1]+RB[0]-LT[0])
            ret.append((LT, RB))
        return ret

    def clearSelection(self):
        self.rects = []
        self.drawRectangles()


class GaussFitDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rawImageWidet = MyImageWidget(self)
        self.fittedImageWidget = MyImageWidget(self)
        self.prmsTable = QTableWidget(1,6,self)
        self.prmsTable.setHorizontalHeaderLabels(["Area", "Amp.", "Cx", "Cy", "Sx", "Sy", "Background"])
        self.leftButton = QToolButton()
        self.leftButton.setArrowType(Qt.LeftArrow)
        self.rightButton = QToolButton()
        self.rightButton.setArrowType(Qt.RightArrow)
        self.leftButton.clicked.connect(lambda: self.changeCurrent(-1))
        self.rightButton.clicked.connect(lambda: self.changeCurrent(1))
        self.acceptButton = QPushButton("Proceed")
        self.rejectButton = QPushButton("Cancel")
        self.acceptButton.clicked.connect(self.accept)
        self.rejectButton.clicked.connect(self.reject)

        imgLayout = QHBoxLayout()
        imgLayout.addWidget(self.leftButton)
        imgLayout.addWidget(self.rawImageWidet)
        imgLayout.addWidget(self.fittedImageWidget)
        imgLayout.addWidget(self.rightButton)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.acceptButton)
        bottomLayout.addWidget(self.rejectButton)

        mainLayout = QVBoxLayout(self)
        mainLayout.addLayout(imgLayout)
        mainLayout.addWidget(self.prmsTable)
        mainLayout.addLayout(bottomLayout)

        self.results = []
        self.current = 0

    def setResults(self):
        num = len(self.results)
        self.prmsTable.clearContents()
        self.prmsTable.setRowCount(1)
        for result, i in zip(self.results, range(num)):
            for val, j in zip(result[2], range(len(result[2]))):
                item = QTableWidgetItem(f"{val:.5g}")
                self.prmsTable.setItem(i, j, item)
            self.prmsTable.insertRow(i+1)

    def resizeImage(self, img):
        img_height, img_width = img.shape
        scale_w = float(396) / float(img_width)
        scale_h = float(396) / float(img_height)
        scale = min([scale_w, scale_h])

        if scale == 0:
            scale = 1

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = cv2.convertScaleAbs(img, alpha=1/16)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        return image

    def showResults(self, num):
        self.current = num
        number = len(self.results)
        result = self.results[num]
        rawImg = self.resizeImage(result[0])
        fittedImg = self.resizeImage(result[1])
        self.rawImageWidet.setImage(rawImg)
        self.fittedImageWidget.setImage(fittedImg)
        self.leftButton.setEnabled(num > 0)
        self.rightButton.setEnabled(num < number-1)

    def changeCurrent(self, drct):
        self.current += drct
        self.showResults(self.current)


class centralWidget(QWidget):

    def __init__(self, parent=None):
        super(centralWidget, self).__init__(parent)

        self.imageAcquirer = ImageAcquirer()
        self.imagePlayer = ImagePlayer()
        self.imageProcessor = ImageProcessor()
        self.ImgWidget = PosLabeledImageWidget(self)
        self.SLM_Controller = SLM_Controller(self)
        self.imageLoader = imageLoader(self)
        self.processWidget = processWidget(self)
        self.DIOWidget = DIOWidget(self)
        self.shutterWidget = shutterWidget(self)
        self.logWidget = LogWidget(self)
        self.tempWidget = TemperatureWidget(self)
        self.piezoWidget = PiezoWidget(self)
        self.acquisitionWidget = AcquisitionWidget(self)

        self.modeTab = QTabWidget()
        self.modeTab.addTab(self.acquisitionWidget, 'Image Acquisition')
        self.modeTab.addTab(self.imageLoader, 'Image Load')

        self.specialDialog = SpecialMeasurementDialog()
        self.gaussFitDialog = GaussFitDialog()

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
        self.imageLoader.gaussFitButton.toggled.connect(self.gaussFitData)

        self.ImgWidget.imageWidget.posSignal.connect(self.writeMousePosition)

        self.processWidget.applyButton.clicked.connect(self.applyProcessClicked)
        self.processWidget.areaSelectButton.toggled.connect(self.selectArea)

        self.DIOWidget.DIObuttons.buttonClicked[int].connect(self.writeDIO)

        self.shutterWidget.shutterButtons.buttonClicked[int].connect(self.writeDIO)

        self.acquisitionWidget.contWidget.markerPositionBoxX.valueChanged.connect(self.moveMarkerX)
        self.acquisitionWidget.contWidget.markerPositionBoxY.valueChanged.connect(self.moveMarkerY)
        self.acquisitionWidget.contWidget.markerFactorBox.valueChanged.connect(self.factorChanged)
        self.acquisitionWidget.contWidget.splitButton.toggled.connect(self.splitMarker)

        self.acquisitionWidget.fixedWidget.specialButton.toggled.connect(self.setupSpecialMeasurement)

        self.acquisitionWidget.initButton.clicked.connect(self.initializeCamera)
        self.acquisitionWidget.finButton.clicked.connect(self.finalizeCamera)
        self.acquisitionWidget.runButton.toggled.connect(self.startAcquisition)
        self.acquisitionWidget.applyButton.clicked.connect(self.applySettings)

        self.SLM_Controller.pitchBox.valueChanged.connect(self.SLM_pitchChanged)
        self.SLM_Controller.SLMDial.valueChanged.connect(self.SLM_dialChanged)

        self.tempWidget.tempButton.clicked.connect(self.showTemperature)
        self.tempWidget.tempStatusButton.clicked.connect(self.showTempStatus)
        self.tempWidget.sensorCoolingButton.toggled.connect(self.sensorCooling)

    def initUI(self):

        self.initLayout()

    def initLayout(self):

        vbox0121 = QVBoxLayout()
        vbox0121.addWidget(self.DIOWidget)
        vbox0121.addWidget(self.shutterWidget)

        vbox0122 = QVBoxLayout()
        vbox0122.addWidget(self.logWidget)
        vbox0122.addWidget(self.tempWidget)

        hbox012 = QHBoxLayout()
        hbox012.addWidget(self.SLM_Controller)
        hbox012.addLayout(vbox0121)
        hbox012.addLayout(vbox0122)

        vbox01 = QVBoxLayout()
        vbox01.addWidget(self.modeTab)
        vbox01.addWidget(self.processWidget)
        vbox01.addLayout(hbox012)

        vbox00 = QVBoxLayout()
        vbox00.addWidget(self.ImgWidget)
        vbox00.addWidget(self.piezoWidget)

        hbox0 = QHBoxLayout(self)
        hbox0.addLayout(vbox00)
        hbox0.addLayout(vbox01)

    def initVal(self):

        self.MarkerX = 0
        self.MarkerY = 0
        self.imgHeight = 600
        self.imgWidth = 600
        self.markerPos = [396, 300]
        self.dst = [0,0,0,0]
        self.split = False
        self.Handle = ct.c_int()
        self.ref = ct.c_int()
        self.CentralPos = None
        self.outDIO = [0, 0, 0, 0, 0, 0]
        self.stopDll = ct.c_bool(False)
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
        logging.info('Successfully connected')
        self.acquisitionWidget.initButton.setEnabled(False)
        self.acquisitionWidget.applyButton.setEnabled(True)
        self.acquisitionWidget.finButton.setEnabled(True)

        self.DIOhandle = ct.c_void_p()
        dll.initDIO(ct.byref(self.DIOhandle))
        self.fin = False

    def finalizeCamera(self):
        dll.fin(self.Handle)
        logging.info('Successfully Disconnected')
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
        size = max([self.acquisitionWidget.AOIWidth, self.acquisitionWidget.AOIHeight])
        vec = np.array([np.cos(self.SLM_Controller.theta+np.pi/4), np.sin(self.SLM_Controller.theta+np.pi/4)])\
        *self.MarkerFactor*600/size/self.SLM_Controller.pitch
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
        scale_w = float(600) / float(img_width)
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

    def setupSpecialMeasurement(self, checked):
        if checked:
            if self.specialDialog.exec_():
                self.specialPrms = self.specialDialog.applySettings()
            else:
                self.acquisitionWidget.fixedWidget.specialButton.setChecked(False)

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

            else:
                if self.acquisitionWidget.fixedWidget.commentButton.isChecked():
                    mainDir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                    with open(mainDir+"log.txt", "w") as f:
                        f.write(self.acquisitionWidget.fixedWidget.comment)
                if self.acquisitionWidget.fixedWidget.specialButton.isChecked():
                    if self.specialPrms["mode"] == 1:
                        mainDir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                        num = self.specialPrms["num"]
                        logging.info('Acquisition start')
                        positions = np.arange(self.specialPrms["start"], self.specialPrms["end"], self.specialPrms["step"])
                        waitTime = time.time()
                        while True:
                            now = time.time()
                            if (now - 300) > waitTime:
                                break
                        for pos in positions:
                            if not self.acquisitionWidget.runButton.isChecked():
                                break
                            dll.movePiezo(self.piezoWidget.piezoID, pos)
                            waitTime = time.time()
                            while True:
                                now = time.time()
                                if (now - 60) > waitTime:
                                    break
                            self.acquisitionWidget.fixedWidget.currentRepeatBox.setText(str(pos))
                            self.imageAcquirer.setup(self.Handle, dir=ct.c_char_p(mainDir),
                                                     num=ct.c_int(num), count_p=count_p, mode=3, piezoID=self.piezoWidget.piezoID)
                            self.imageAcquirer.start()
                            ref = count.value
                            while not self.imageAcquirer.stopped:
                                if count.value > (ref + 5):
                                    self.acquisitionWidget.fixedWidget.countBox.setText(str(count.value))
                                    QApplication.processEvents()
                                    ref = count.value
                            count = ct.c_int(0)
                            count_p = ct.pointer(count)
                        self.acquisitionWidget.runButton.setChecked(False)
                        logging.info('Acquisition stopped')
                    elif self.specialPrms["mode"] == 2:
                        mainDir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                        num = self.specialPrms["num"]
                        logging.info('Acquisition start')
                        positions = np.arange(self.specialPrms["start"], self.specialPrms["end"], self.specialPrms["step"])
                        XY = self.specialPrms["tiltXY"]
                        for pos in positions:
                            if XY == 0:
                                self.SLM_Controller.tiltXChanged(pos)
                            else:
                                self.SLM_Controller.tiltYChanged(pos)
                            waitTime = time.time()
                            while True:
                                now = time.time()
                                if (now - 3) > waitTime:
                                    break
                            self.acquisitionWidget.fixedWidget.currentRepeatBox.setText(str(pos))
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
                        self.acquisitionWidget.runButton.setChecked(False)
                        logging.info('Acquisition stopped')
                    elif self.specialPrms["repeatCheck"]:
                        mainDir = self.acquisitionWidget.fixedWidget.dirname.encode(encoding='utf_8')
                        num = int(self.acquisitionWidget.fixedWidget.numImgBox.text())
                        repeat = self.specialPrms["repeat"]
                        self.shutterWidget.closeButton.click()
                        QApplication.processEvents()
                        logging.info('Acquisition start')
                        for i in range(repeat*2):
                            self.acquisitionWidget.fixedWidget.currentRepeatBox.setText("Measurement "+str(i%2+1)+" / Cycle "+str(i//2+1))
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
                            self.shutterWidget.change()
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
        else:
            self.imageAcquirer.stopDll.value = True
            self.imageAcquirer.stopped = True
            self.imagePlayer.stopped = True
            self.acquisitionWidget.runButton.setText("RUN")

    def acquisitionFinished(self):
        self.acquisitionWidget.runButton.setChecked(False)

    def applyCenter(self, posX, posY):
        self.CentralPos = (posX, posY)
        self.acquisitionWidget.runButton.setChecked(False)
        self.acquisitionWidget.fixedWidget.cntrPosBox.setText("{:.2f}, {:.2f}".format(self.CentralPos[0], self.CentralPos[1]))

    def setAOICenter(self):
        centerX = self.acquisitionWidget.AOI.CenterXBox.value()
        centerY = self.acquisitionWidget.AOI.CenterYBox.value()
        AOISizeIndex = self.acquisitionWidget.AOI.AOISizeBox.currentIndex()
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
        isDefault = self.acquisitionWidget.AOI.defaultCheck.isChecked()
        if isDefault:
            index = self.acquisitionWidget.AOI.AOISizeBox.currentIndex()
            val = 2048/(2**index)
            self.acquisitionWidget.AOIWidth = val
            self.acquisitionWidget.AOIHeight = val
            if index == 0:
                if dll.SetInt(self.Handle, "AOIWidth", 2048):
                    logging.error("AOIWidth")
                if dll.SetInt(self.Handle, "AOIHeight", 2048):
                    logging.error("AOIHeight")
            else:
                self.setAOICenter()
        else:
            self.acquisitionWidget.AOIWidth = self.acquisitionWidget.AOI.AOIWidthBox.value()
            self.acquisitionWidget.AOIHeight = self.acquisitionWidget.AOI.AOIHeightBox.value()
            if dll.SetInt(self.Handle, "AOIWidth", self.acquisitionWidget.AOI.AOIWidthBox.value()):
                logging.error("AOIWidth")
            if dll.SetInt(self.Handle, "AOILeft", self.acquisitionWidget.AOI.AOILeftBox.value()):
                logging.error("AOILeft")
            if dll.SetInt(self.Handle, "AOIHeight", self.acquisitionWidget.AOI.AOIHeightBox.value()):
                logging.error("AOIHeight")
            if dll.SetInt(self.Handle, "AOITop", self.acquisitionWidget.AOI.AOITopBox.value()):
                logging.error("AOITop")

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
        self.imageAcquirer.areaIsSelected = self.processWidget.areaIsSelected
        self.imageAcquirer.selectedAreas = self.processWidget.selectedAreas
        self.imageLoader.prms = self.processWidget.prmStruct
        self.imageLoader.areaIsSelected = self.processWidget.areaIsSelected
        self.imageLoader.selectedAreas = self.processWidget.selectedAreas
        if update:
            self.imageLoader.update_img()

    def writeMousePosition(self, pos):
        if self.ImgWidget.imageWidget.image is not None:
            x = pos.x()
            y = pos.y()
            w = self.ImgWidget.imageWidget.image.size().width()
            h = self.ImgWidget.imageWidget.image.size().height()
            realX = x/w*self.acquisitionWidget.AOIWidth
            realY = y/h*self.acquisitionWidget.AOIHeight
            self.ImgWidget.writeLabel(f"Position (pix.): [{realX:.3f}, {realY:.3f}]")

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
        dll.writeDIO(self.DIOhandle, ct.c_ubyte(out))

    def gaussFitData(self, checked):
        if checked:
            width, height, stride = self.imageLoader.width, self.imageLoader.height, self.imageLoader.stride
            ImgArry = (ct.c_ushort * (width * height))()
            prmsList = []
            datfile, imgSize = self.imageLoader.dirname+"/spool.dat", self.imageLoader.imgSize
            mm = self.imageLoader.mm
            mm.seek(imgSize*0)
            rawdata = mm.read(imgSize)
            buffer = ct.cast(rawdata, ct.POINTER(ct.c_ubyte))
            ret = dll.convertBuffer(buffer, ImgArry, width, height, stride)
            imgNpArray = np.array(ImgArry).reshape(width, height)
            areaResults = []
            for area, i  in zip(self.imageLoader.selectedAreas, range(len(self.imageLoader.selectedAreas))):
                LT, RB = area
                areaWidth = RB[0] - LT[0]
                areaHeight = RB[1] - LT[1]
                areaImg = imgNpArray[LT[1]:RB[1], LT[0]:RB[0]]
                gaussfit = gf.GaussFit(areaImg)
                try:
                    fitted, prms, cov = gaussfit.fit()
                except RuntimeError:
                    logging.warning("fitting failed at area "+str(i))
                else:
                    prms = np.insert(prms, 0, i)
                    areaResults.append([areaImg, fitted, prms, cov])
            if areaResults:
                self.gaussFitDialog.results = areaResults
                self.gaussFitDialog.setResults()
                self.gaussFitDialog.showResults(0)
                if not self.gaussFitDialog.exec_():
                    self.imageLoader.gaussFitButton.setChecked(False)
            else:
                self.imageLoader.gaussFitButton.setChecked(False)
                logging.warning("No results to show")

    def analyzeDatas(self):
        self.applyProcessSettings()
        logging.info("applySettings")
        start = self.imageLoader.anlzStartBox.value()
        end = self.imageLoader.anlzEndBox.value()
        if self.imageLoader.old:
            processfiles = []
            for i in range(start, end):
                processfiles.append(self.imageLoader.datfiles[i])
            logging.info("setup")
            self.imageProcessor.setup(processfiles, self.imageLoader.width,
                                      self.imageLoader.height, self.imageLoader.stride,
                                      self.processWidget.prmStruct, self.processWidget.areaIsSelected, self.processWidget.selectedAreas,
                                      self.imageLoader.dirname, self.imageLoader.progressBar, self.imageLoader.old)
            self.imageProcessor.run()
        else:
            if self.imageLoader.anlzCheck.isChecked():
                filterText = self.imageLoader.anlzFilter.text()
                filterDirs = glob(filterText)
                for dir in filterDirs:
                    datFile = dir +"/spool.dat"
                    with open(datFile, mode="r+b") as f:
                        with mmap.mmap(f.fileno(), 0) as mm:
                            processData = (datFile, start, end, self.imageLoader.imgSize)
                            logging.info("setup")
                            self.imageProcessor.setup(processData, self.imageLoader.width,
                                                      self.imageLoader.height, self.imageLoader.stride,
                                                      self.processWidget.prmStruct, self.imageLoader.gaussFitButton.isChecked(), self.processWidget.selectedAreas,
                                                      dir, self.imageLoader.progressBar, self.imageLoader.old, mm)
                            self.imageProcessor.run()
                            while not self.imageProcessor.stopped:
                                pass

            else:
                datFile = self.imageLoader.dirname +"/spool.dat"
                processData = (datFile, start, end, self.imageLoader.imgSize)
                logging.info("setup")
                self.imageProcessor.setup(processData, self.imageLoader.width,
                                          self.imageLoader.height, self.imageLoader.stride,
                                          self.processWidget.prmStruct, self.imageLoader.gaussFitButton.isChecked(), self.processWidget.selectedAreas,
                                          self.imageLoader.dirname, self.imageLoader.progressBar, self.imageLoader.old, self.imageLoader.mm)
                self.imageProcessor.run()

    def exportBMP(self):
        fileToSave = QFileDialog.getSaveFileName(self, 'File to save', filter="Images (*.png *.bmp *.jpg)")
        logging.info("Save picture: "+fileToSave[0])
        if fileToSave[0]:
            if self.modeTab.currentIndex() == 0:
                img = self.imageAcquirer.img
            else:
                img = self.imageLoader.img
            cv2.imwrite(fileToSave[0], img)

    def exportSeries(self):
        dirToSave = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dirToSave:
            start = self.imageLoader.anlzStartBox.value()
            end = self.imageLoader.anlzEndBox.value()
            if self.imageLoader.old:
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
            else:
                ImgArry = (ct.c_ushort * (self.imageLoader.width * self.imageLoader.height))()
                outBuffer = (ct.c_ubyte*(self.imageLoader.width*self.imageLoader.height))()
                num = end - start
                self.imageLoader.mm.seek(self.imageLoader.imgSize*start)
                for i in range(num):
                    rawdata = self.imageLoader.mm.read(int(self.imageLoader.imgSize))
                    buffer = ct.cast(rawdata, ct.POINTER(ct.c_ubyte))
                    max = ct.c_double()
                    min = ct.c_double()
                    ret = dll.convertBuffer(buffer, ImgArry, self.imageLoader.width, self.imageLoader.height, self.imageLoader.stride)
                    dll.processImageShow(self.imageLoader.height, self.imageLoader.width, ImgArry, self.imageLoader.prms, outBuffer,
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
        self.acquisitionWidget.AOI.AOILeftBox.setValue(settings.value('AOI left', type=int))
        self.acquisitionWidget.AOI.AOITopBox.setValue(settings.value('AOI top', type=int))
        self.acquisitionWidget.AOI.AOIWidthBox.setValue(settings.value('AOI width', type=int))
        self.acquisitionWidget.AOI.AOIHeightBox.setValue(settings.value('AOI height', type=int))
        self.acquisitionWidget.AOI.CenterXBox.setValue(settings.value('Center X', type=int))
        self.acquisitionWidget.AOI.CenterYBox.setValue(settings.value('Center Y', type=int))
        self.acquisitionWidget.fixedWidget.thresBox.setValue(settings.value('DIO threshold', type=float))
        self.acquisitionWidget.contWidget.markerFactorBox.setValue(settings.value('marker factor', 100, type=float))
        self.imageLoader.dirname = settings.value('dir image', '')
        self.SLM_Controller.wavelengthBox.setCurrentIndex(settings.value('SLM wavelength', type=int))
        self.SLM_Controller.pitchBox.setValue(settings.value('SLM pitch', 23, type=float))
        self.SLM_Controller.focusBox.setValue(settings.value('SLM focus', 0, type=float))
        self.SLM_Controller.focusXBox.setValue(settings.value('SLM focusX', 0, type=int))
        self.SLM_Controller.focusYBox.setValue(settings.value('SLM focusY', 0, type=int))
        self.SLM_Controller.aspectBox.setValue(settings.value('SLM aspect', 1, type=float))

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
        self.settings.setValue('AOI left', self.acquisitionWidget.AOI.AOILeftBox.value())
        self.settings.setValue('AOI top', self.acquisitionWidget.AOI.AOITopBox.value())
        self.settings.setValue('AOI width', self.acquisitionWidget.AOI.AOIWidthBox.value())
        self.settings.setValue('AOI height', self.acquisitionWidget.AOI.AOIHeightBox.value())
        self.settings.setValue('Center X', self.acquisitionWidget.AOI.CenterXBox.value())
        self.settings.setValue('Center Y', self.acquisitionWidget.AOI.CenterYBox.value())
        self.settings.setValue('DIO threshold', self.acquisitionWidget.fixedWidget.thresBox.value())
        self.settings.setValue('marker factor', self.acquisitionWidget.contWidget.markerFactorBox.value())
        self.settings.setValue('dir image', self.imageLoader.dirname)
        self.settings.setValue('SLM wavelength', self.SLM_Controller.wavelengthBox.currentIndex())
        self.settings.setValue('SLM pitch', self.SLM_Controller.pitchBox.value())
        self.settings.setValue('SLM focus', self.SLM_Controller.focusBox.value())
        self.settings.setValue('SLM focusX', self.SLM_Controller.focusXBox.value())
        self.settings.setValue('SLM focusY', self.SLM_Controller.focusYBox.value())
        self.settings.setValue('SLM aspect', self.SLM_Controller.aspectBox.value())

    def showTemperature(self):
        temp = ct.c_double()
        dll.GetFloat(self.Handle, "SensorTemperature", ct.byref(temp))
        logging.info(temp.value)

    def showTempStatus(self):
        status = ct.c_int()
        dll.GetEnumIndex(self.Handle, "TemperatureStatus", ct.byref(status))
        statusList = ["Cooler Off", "Stabilised", "Cooling", "Drift", "Not Stabilized", "Fault"]
        logging.info(f"Temperature Status: "+ statusList[status.value])

    def sensorCooling(self, isChecked):
        dll.SetBool(self.Handle, "SensorCooling", isChecked)

    def selectArea(self, checked):
        if checked:
            try:
                if self.modeTab.currentIndex() == 0:
                    img = self.imageAcquirer.img
                else:
                    img = self.imageLoader.img
                self.areaDialog = AreaSelectDialog(img)
                if self.areaDialog.exec_():
                    self.processWidget.selectedAreas = self.areaDialog.applyAreas()
                else:
                    self.processWidget.areaSelectButton.setChecked(False)
            except AttributeError:
                self.processWidget.areaSelectButton.setChecked(False)
                logging.error("No Image to draw for select areas")


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

        self.setWindowTitle('Andor_CMOS')
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)

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
