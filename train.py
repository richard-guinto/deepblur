#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:10:45 2018

@author: richard
"""
import numpy as np
import os.path
import sys


from keras.applications import mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import ModelCheckpoint


def main():
    
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [existing model.h5]'.format(name))
        exit(1)


    base_model = mobilenet.MobileNet(weights='imagenet', include_top = False, input_shape=(224,224,3))
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation    
    
    new_model=Model(inputs=base_model.input,outputs=preds)
    #specify the inputs
    #specify the outputs
    #now a model has been created based on our architecture

    #for i,layer in enumerate(new_model.layers):
    #    print(i,layer.name)

    for layer in new_model.layers[:20]:
        layer.trainable=False
        
    for layer in new_model.layers[20:]:
        layer.trainable=True
        
    if len(sys.argv) == 2:
        new_model.load_weights(sys.argv[1])

    
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 
    #included in our dependencies
    
    train_dir = "/datasets/DEEPBLUR/Train"
    valid_dir = "/datasets/DEEPBLUR/Valid"

    train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 class_mode='categorical',
                                                 shuffle=True)
    
    valid_generator=train_datagen.flow_from_directory(valid_dir,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 class_mode='categorical',
                                                 shuffle=True)
    
    
    
    new_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    new_model.summary()
    
    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.{epoch:02d}.h5'), save_weights_only=True)


    step_size_train=train_generator.n//train_generator.batch_size
    step_size_valid=valid_generator.n//valid_generator.batch_size
    new_model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = valid_generator,
                   validation_steps = step_size_valid,
                   epochs=5,
                   callbacks=[checkpoint])
    
    new_model.evaluate_generator(generator=valid_generator)
    
    
if __name__ == '__main__':
    main()
    
