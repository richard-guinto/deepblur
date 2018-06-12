# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:38:44 2018

@author: rf.guinto
"""

import cv2
import numpy as np
import os.path
import glob

#default_path = '/datasets/CELABA/img_celeba'
default_path = '../MSCOCO/test2017'
output_dir = 'output'
np.random.seed(3)

def loader(batch_size = 8, path = default_path):
    images = []
    counter = 0
    while True:
        files = glob.glob(os.path.join(path, '*.jpg'))
        for file in files:
            img = cv2.imread(file)
            ksize = np.random.randint(5,20)
            # generating the kernel
            kernel_motion_blur = np.zeros((ksize, ksize))
            kernel_motion_blur[int((ksize-1)/2), :] = np.ones(ksize)
            kernel_motion_blur = kernel_motion_blur / ksize
            # applying the kernel to the input image
            output = cv2.filter2D(img, -1, kernel_motion_blur)
            print('file: %s ksize %d' % (file, ksize))
            images.append(output)
            counter = counter + 1
            output_name = output_dir + '/output' + str(counter) + '_' + str(ksize) + '.jpg'
            cv2.imwrite(output_name, output)
            if len(images) == batch_size:
                yield images
                images.clear()

def main():
    generator = loader()
    imgs = next(generator)
    print(len(imgs))
    imgs = next(generator)
    

if __name__ == '__main__':
    main()
