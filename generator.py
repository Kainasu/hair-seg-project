import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def create_training_generators(dataset = None, augmentation=True):
    """return generator containing training and validation generators to fit model"""

    if dataset is None:
        dataset = 'data/Figaro1k'

    train_dir = os.path.join(dataset, 'Training')

    if augmentation :        
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
    else :
        data_gen_args = dict(rescale = 1.0/255.0,
                    validation_split=0.2
        )

    image_args = dict(color_mode = "rgb",
        target_size=(128,128),
        class_mode=None,
        batch_size=32,
        seed=42
    )

    mask_args = dict(target_size=(128, 128), 
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

    return train_generator, val_generator, len(image_train_generator), len(image_val_generator)

def create_testing_generator(dataset = None):
    if dataset is None:
        dataset = 'data/Figaro1k'
    
    test_dir = os.path.join(dataset, 'Testing')
    # Generator for test
    image_test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
    image_test_generator = image_test_datagen.flow_from_directory(
    os.path.join(test_dir, 'imgs'),
    color_mode = "rgb",
    target_size=(128,128),
    class_mode=None,
    shuffle=False,
    batch_size=1)

    # Generator for test
    mask_test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
    mask_test_generator = mask_test_datagen.flow_from_directory(
    os.path.join(test_dir, 'masks'),
    color_mode = "grayscale",
    target_size=(128,128),
    class_mode=None,
    shuffle=False,
    batch_size=1)

    test_generator = zip(image_test_generator, mask_test_generator)

    return test_generator, len(image_test_generator)