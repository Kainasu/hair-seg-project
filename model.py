from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU

def conv_block(inputs, filters):
    x = Conv2D(filters, (3,3) , activation='relu', padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    return x

def deconv_block(inputs, filters, skip_connection):
    x = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs), skip_connection], axis=3)
    x = conv_block(x, filters)
    return x

def create_unet(image_size=(128,128,3)):
    """returns unet model"""

    inputs = Input(image_size)

    #Downsampling
    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 512)

    #Upsampling
    dconv1 = deconv_block(conv5, 256, conv4)
    dconv2 = deconv_block(dconv1, 128, conv3)
    dconv3 = deconv_block(dconv2, 64, conv2)
    dconv4 = deconv_block(dconv3, 32, conv1)

    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(dconv4)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU(target_class_ids=[1])])

    return model
    
if __name__ == '__main__':
    model = create_unet()
    model.summary()
