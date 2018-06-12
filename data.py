# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:38:44 2018

@author: rf.guinto
"""

import cv2
import numpy as np
import os.path
import glob

default_path = '/datasets/CELABA/img_celeba'

def loader(batch_size = 64, path = default_path):
    while True:
        files = glob.glob(os.path.join(path, '*.jpg'))
        for file in files:
            img = cv2.imread(file)
            yield img
            