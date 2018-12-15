#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:25:53 2018

@author: richard
"""
import numpy as np
import os.path
import sys
import pandas as pd

from keras.models import load_model
from keras.applications import mobilenet
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.h5]'.format(name))
        exit(1)

    if len(sys.argv) == 2:
        print('loading model ', sys.argv[1])
        #base_model = load_model(sys.argv[1], compile=False)
        base_model = mobilenet.MobileNet(weights=None, include_top = False, input_shape=(224,224,3))
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dense(512,activation='relu')(x) #dense layer 3
        preds=Dense(2,activation='softmax')(x) #final layer with softmax activation    
        new_model=Model(inputs=base_model.input,outputs=preds)
        new_model.load_weights(sys.argv[1])
        model = new_model
        model.summary()
    else:
        print('using Mobilenet model')
        base_model = mobilenet.MobileNet(weights='imagenet', include_top = False, input_shape=(224,224,3))
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dense(512,activation='relu')(x) #dense layer 3
        preds=Dense(2,activation='softmax')(x) #final layer with softmax activation    
        new_model=Model(inputs=base_model.input,outputs=preds)
        model = new_model
        model.summary()

    train_dir = "/datasets/DEEPBLUR/Train"
    test_dir = "/datasets/DEEPBLUR/Test"
    
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 class_mode='categorical',
                                                 shuffle=True)
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator=test_datagen.flow_from_directory(test_dir,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode=None,
                                                 shuffle=False)

    print('resetting generator')
    test_generator.reset()
    print('predicting...')
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    pred=model.predict_generator(test_generator,verbose=1)
    print(pred)
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    results.to_csv("results.csv",index=False)


if __name__ == '__main__':
    main()
