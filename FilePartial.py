import mmap
import ctypes as ct
import numpy as np
import os
import socket
import matplotlib.pyplot as plt

<<<<<<< HEAD
# dir = r"C:\Users\Ryoma\Documents\00_Simulation\program_test\23_41_58"
dir = "C:\Users\Shimura-lab\Documents\Fukuhara\Experiment\test\23_41_58"
=======
if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

simMode = True
if socket.gethostname() == "EXPERIMENT2":
    simMode = False
    os.chdir(r'../Andor_dll/Andor_dll')
else:
    os.chdir(r'Andor_dll\Andor_dll')

dll = ct.windll.LoadLibrary(r'..\x64\Release\Andor_dll.dll')
dll.InitialiseUtilityLibrary()
dir = r"C:\Users\Ryoma\Documents\00_Simulation\program_test\23_41_58"
>>>>>>> 69bdb06f6c1c54bd4d98bfdcb27cda81590b30df
metafile = dir + "/metaSpool.txt"
file = dir + "/movie.dat"
dll.convertBuffer.argtypes = [ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_ushort), ct.c_longlong, ct.c_longlong, ct.c_longlong]
with open(metafile, mode='r') as f:
    metadata = f.readlines()

size, code, stride, height, width, frate = metadata
size, stride, height, width, frate = map(int, [size, stride, height, width, frate])

with open(file, mode='r+b') as f:
    mm = mmap.mmap(f.fileno(), 0)
    mm.seek(size*2)
    img = mm.read(int(size))
<<<<<<< HEAD
    print(type(img))
=======
    buffer = ct.cast(img, ct.POINTER(ct.c_ubyte))
    outputBuffer = (ct.c_ushort * (width * height))()
    ret = dll.convertBuffer(buffer, outputBuffer, width, height, stride)
    outimg = np.array(outputBuffer).reshape(height, width)
    plt.imshow(outimg)
    plt.show()
>>>>>>> 69bdb06f6c1c54bd4d98bfdcb27cda81590b30df
    mm.close()

dll.FinaliseUtilityLibrary()
