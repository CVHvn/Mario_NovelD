import numpy as np
import cv2
import math
from collections import defaultdict

class SimplePixelHash:
    def __init__(self, resize=42):
        self.resize = resize
        self.table = defaultdict(int)

    def hash(self, frame):
        # 1. resize small
        gray = cv2.resize(frame, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        # 2. threshold to binary
        bits = (frame > 128).astype(np.uint8).flatten()
        # 3. pack bits into integer hash
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def update(self, frame):
        key = self.hash(frame)
        self.table[key] += 1
        count = self.table[key]
        return count

    def count(self, frame):
        key = self.hash(frame)
        count = self.table[key]
        return count

    def reset(self):
        self.table = defaultdict(int)