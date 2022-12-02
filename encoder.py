from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU

def create_unet(image_size=(128,128,3)):
    """returns unet model"""

    inputs = Input(image_size)

    #Downsampling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    # poo5a = MaxPooling2D(pool_size=(2, 2))(conv5)
    # conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(poo5a)
    # conv5a = BatchNormalization()(conv5a)
    # conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5a)
    # conv5a = BatchNormalization()(conv5a)

    #Upsampling
    # up6a = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5a), conv5], axis=3)
    # conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(up6a)
    # conv6a = BatchNormalization()(conv6a)
    # conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6a)
    # conv6a = BatchNormalization()(conv6a)

    # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3) #Was conv5 instead of conv6a
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)
    # up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)
    # up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)
    # up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv10 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)

    model = Model(inputs=[inputs], outputs=[conv5])

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc'])

    return model
    
if __name__ == '__main__':
    model = create_unet()
    model.summary()
