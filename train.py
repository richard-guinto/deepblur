#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:10:45 2018

@author: richard
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from data import loader

image_dim = (100,100,3)


def main():
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', input_shape=image_dim))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit_generator(generator=loader)
    
    
if __name__ == '__main__':
    main()
    