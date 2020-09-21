import mmap

dir = r"C:\Users\Ryoma\Documents\00_Simulation\program_test\23_41_58"
metafile = dir + "/metaSpool.txt"
file = dir + "/movie.dat"
with open(metafile, mode='r') as f:
    metadata = f.readlines()

size, code, stride, height, width, frate = metadata

with open(file, mode='r+b') as f:
    mm = mmap.mmap(f.fileno(), 0)
    img = mm.read(int(size))
    print(img)
    mm.close()
