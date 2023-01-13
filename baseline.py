import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_backgroundBaseline(image_size=(128,128,3)):    
    # Create the model
    inputs = Input(shape=image_size)
    output = Lambda(lambda x: K.zeros_like(x[..., 0]))(inputs)    
    model = Model(inputs, output)

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU(target_class_ids=[1])])

    return model

def create_hairBaseline(image_size=(128,128,3)):    
    # Create the model
    inputs = Input(shape=image_size)
    output = Lambda(lambda x: K.ones_like(x[...,0]))(inputs)
    model = Model(inputs, output)

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU(target_class_ids=[1])])

    return model

if __name__ == '__main__':

    backgroundBaseline_path = 'models/baseline/onlyBlack'
    os.makedirs(backgroundBaseline_path, exist_ok=True)
    whitegroundBaseline_path = 'models/baseline/onlyWhite'
    os.makedirs(whitegroundBaseline_path, exist_ok=True)

    baseline1 = create_backgroundBaseline((128,128,3))
    baseline1.save(os.path.join(backgroundBaseline_path, 'model.h5'))

    baseline2 = create_hairBaseline((128,128,3))
    baseline2.save(os.path.join(whitegroundBaseline_path, 'model.h5'))
