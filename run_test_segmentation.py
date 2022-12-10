from generator import create_testing_generator, create_training_generators
import os
import numpy as np
from model import create_unet
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()        
    parser.add_argument('--model', dest='model', action='store', required=True)    
    args = parser.parse_args()

    root_dir = './'
    model = load_model(args.model)
    train_generator, val_generator, train_steps, val_steps = create_training_generators()
    test_generator, test_steps = create_testing_generator()
    score = model.evaluate(train_generator, steps=train_steps)
    print(f'Training ==> acc: {score[1]}')
    score = model.evaluate(test_generator, steps=test_steps)
    print(f'Testing ==> acc: {score[1]}')
    
    
    