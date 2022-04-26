import mmap
import time
import numpy as np
import cv2 as cv
import os

class mmapManager:
    # file_name : path to file
    def __init__(self, file, size, shape = (0, 0, 0)) -> None:
        self._mm = None
        self._file = file
        self._size = size
        self._shape = shape

        if not os.path.exists(file):
            self._createMMapFile()

        self._readMMapFile()

    def _createMMapFile(self):
        with open(self._file, 'wb') as f:
            initStr = '00' * self._size
            initByte = bytes.fromhex(initStr)
            f.write(initByte)
    
    def _readMMapFile(self):
        with open(self._file, 'r+b') as f:
            self._mm = mmap.mmap(f.fileno(), 0)

    def ReadImage(self):
        self._mm.seek(0)
        buf = self._mm.read(self._size)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(self._shape)
        return img
    
    # image : cv image (h*w*c 3D ndarray)
    def WriteImage(self, image):
        self._mm.seek(0)
        buf = image.tobytes()
        self._mm.write(buf)
        self._mm.flush()

    def ReadString(self):
        self._mm.seek(0)
        return self._mm.read().decode('utf-8')

    # words : str object to sent
    def WriteString(self, words):
        self._mm.seek(0)
        self._mm.write(words.encode('utf-8'))
        self._mm.flush()

    def dispose(self):
        self._mm.close()
