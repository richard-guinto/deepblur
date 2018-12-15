# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:38:44 2018

@author: rf.guinto
"""

import cv2
import numpy as np
import os.path
import glob
from pathlib import Path

default_path = '/datasets/MSCOCO/train2017'
#output_dir = '/datasets/DEEPBLUR/Train'
train_dir = '/datasets/DEEPBLUR/Train/motion'
valid_dir = '/datasets/DEEPBLUR/Valid/motion'
train_dir_orig = '/datasets/DEEPBLUR/Train/nonblur'
valid_dir_orig = '/datasets/DEEPBLUR/Valid/nonblur'
np.random.seed(3)

def loader(batch_size = 8, path = default_path, output_dir = 'blur', orig_dir = 'nonblur'):
    images = []
    counter = 0
    while True:
        files = glob.glob(os.path.join(path, '*.jpg'))
        for file in files:
            img = cv2.imread(file)
            # crop the image to (224,224)
            #img = img[0:224,0:224]
            ksize = np.random.randint(5,20)
            # generating the kernel
            kernel_motion_blur = np.zeros((ksize, ksize))
            kernel_motion_blur[int((ksize-1)/2), :] = np.ones(ksize)
            kernel_motion_blur = kernel_motion_blur / ksize
            # applying the kernel to the input image
            output = cv2.filter2D(img, -1, kernel_motion_blur)
            #print('file: %s ksize %d' % (file, ksize))
            images.append(output)
            counter = counter + 1
            output_name = output_dir + '/motion' + str(counter) + '_' + str(ksize) + '.jpg'
            cv2.imwrite(output_name, output)
            output_name = orig_dir + '/' + Path(file).name
            cv2.imwrite(output_name, img)
            if len(images) == batch_size:
                yield images
                images.clear()

def main():
    
    generator = loader(output_dir = train_dir, orig_dir = train_dir_orig)
    for i in range(1000):
    	imgs = next(generator)
    
    generator = loader(output_dir = valid_dir, orig_dir = valid_dir_orig)
    for i in range(300):
    	imgs = next(generator)


if __name__ == '__main__':
    main()

