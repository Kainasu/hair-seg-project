import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def create_train_generator(train_dir = None):
    """return generator containing training and validation generators to fit model"""

    if train_dir is None:
        train_dir = './Figaro1k/Training'

    # data augmentation
    data_gen_args = dict(featurewise_center=False,
        rescale = 1.0/255.0,
        fill_mode = 'constant',
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    image_args = dict(color_mode = "rgb",
        target_size=(128,128),
        class_mode=None,
        batch_size=32,
        seed=42
    )

    mask_args = dict( target_size=(128, 128), 
        batch_size=32,
        color_mode = "grayscale",
        interpolation = "nearest",  
        class_mode=None,
        seed=42
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    
    image_train_generator = image_datagen.flow_from_directory(
        os.path.join(train_dir, 'imgs'),
        subset='training',
        **image_args)

    mask_train_generator = mask_datagen.flow_from_directory(
        os.path.join(train_dir, 'masks'),               
        subset='training',
        **mask_args)
                
    image_val_generator = image_datagen.flow_from_directory(
        os.path.join(train_dir, 'imgs'),
        subset='validation',
        **image_args)

    mask_val_generator = mask_datagen.flow_from_directory(
        os.path.join(train_dir, 'masks'),
        subset='validation',
        **mask_args)


    # combine generators into one which yields image and masks
    train_generator = zip(image_train_generator, mask_train_generator)
    val_generator = zip(image_val_generator, mask_val_generator)

    return train_generator, val_generator