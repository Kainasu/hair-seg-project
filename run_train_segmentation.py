from generator import create_training_generators
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

if __name__ == '__main__':

    root_dir = './'
    model = create_unet()
    train_generator, val_generator, train_steps, val_steps = create_training_generators()
    #Create directory to save model and history
    num_train = 0
    while os.path.isdir(os.path.join(root_dir, 'training_{i}'.format(i=num_train))):
        num_train +=1
    save_dir = os.path.join(root_dir, 'training_{i}'.format(i=num_train))
    os.makedirs(save_dir)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)

    #train model
    epochs = 50
    history = model.fit(train_generator,validation_data=val_generator,epochs=epochs,batch_size=32, steps_per_epoch=train_steps, validation_steps=val_steps, callbacks=[reduce_lr])
    #history = model.fit(train_generator,epochs=epochs,batch_size=64, steps_per_epoch=len(image_train_generator))
    model.save(os.path.join(save_dir, 'model.h5'))
    np.save(os.path.join(save_dir, 'my_history.npy') ,history)