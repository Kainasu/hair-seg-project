import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU
import matplotlib.pyplot as plt
import argparse

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

    parser = argparse.ArgumentParser()

    # Add argument for input size
    parser.add_argument('--size', type=int,
                    help='Input size', default=128)

    args = parser.parse_args()
    
    image_size = (args.size, args.size, 3)

    backgroundBaseline_path = f'models/baseline/onlyBlack/{args.size}x{args.size}'
    os.makedirs(backgroundBaseline_path, exist_ok=True)
    whitegroundBaseline_path = f'models/baseline/onlyWhite/{args.size}x{args.size}'
    os.makedirs(whitegroundBaseline_path, exist_ok=True)

    baseline1 = create_backgroundBaseline(image_size=image_size)
    baseline1.save(os.path.join(backgroundBaseline_path, 'model.h5'))

    baseline2 = create_hairBaseline(image_size=image_size)
    baseline2.save(os.path.join(whitegroundBaseline_path, 'model.h5'))
