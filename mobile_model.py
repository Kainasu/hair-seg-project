from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout, Add, DepthwiseConv2D, Activation
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU

def bottleneck_block(inputs, filters, strides=(1,1), expansion_factor=6, resi=False):
    expand = inputs.shape[-1] * expansion_factor    
    m = Conv2D(expand, (1,1), padding='same')(inputs)
    m = BatchNormalization()(m)
    m = Activation('relu6')(m)
    m = DepthwiseConv2D((3,3), strides=strides, padding='same')(m)
    m = BatchNormalization()(m)
    m = Activation('relu6')(m)    
    m = Conv2D(filters, (1,1), padding='same')(m)
    m = BatchNormalization()(m)
    if resi:    
        return Add()([m, inputs])
    return m



def inverted_residual_block(inputs, filters, strides=(1,1), expansion_factor=6, n=1):
    m = bottleneck_block(inputs, filters, strides, expansion_factor, False)
    for i in range(1, n):
        m = bottleneck_block(m, filters, (1,1), expansion_factor, True)
    return m
    
# Architecture inspired from https://pdfs.semanticscholar.org/44e1/3ee0ad72e84c13635a9716bb7a27f39d27a2.pdf
def create_mobile_unet(image_size=(128,128,3)):
    """returns unet model"""

    inputs = Input(image_size)
    
    #Encoder
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)

    bottleneck1 = inverted_residual_block(conv1, 16, strides=(1,1), expansion_factor=1)

    bottleneck2 = inverted_residual_block(bottleneck1, 24, strides=(2,2), expansion_factor=6, n=2)

    bottleneck3 = inverted_residual_block(bottleneck2, 32, strides=(2,2), expansion_factor=6, n=3)

    bottleneck4 = inverted_residual_block(bottleneck3, 64, strides=(2,2), expansion_factor=6, n=4)

    bottleneck5 = inverted_residual_block(bottleneck4, 96, strides=(1,1), expansion_factor=6, n =3)

    bottleneck6 = inverted_residual_block(bottleneck5, 160, strides=(2,2), expansion_factor=6, n=3)

    bottleneck7 = inverted_residual_block(bottleneck6, 320, strides=(1,1), expansion_factor=6)

    conv2 = Conv2D(1280, (1,1), activation='relu', padding='same')(bottleneck7)
    conv2 = BatchNormalization()(conv2)
    
    #Decoder

    up6 = concatenate([Conv2DTranspose(320, (2, 2), strides=(1, 1), padding='same')(conv2), bottleneck7], axis=3) #Was conv5 instead of conv6a
    conv6 = Conv2D(320, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(320, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = concatenate([Conv2DTranspose(160, (2, 2), strides=(1, 1), padding='same')(conv6), bottleneck6], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = concatenate([Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same')(conv7), bottleneck5], axis=3)
    conv8 = Conv2D(96, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(96, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='same')(conv8), bottleneck4], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    up10 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9), bottleneck3], axis=3)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(up10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = BatchNormalization()(conv10)
    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), bottleneck2], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv11)

    model = Model(inputs=[inputs], outputs=[conv12])



    # d_bottleneck1 = inverted_residual_block(conv2, 96, strides=(2,2), expansion_factor=6)

    # d_bottleneck2 = inverted_residual_block(d_bottleneck1, 32, strides=(2,2), expansion_factor=6)

    # d_bottleneck3 = inverted_residual_block(d_bottleneck2, 24, strides=(2,2), expansion_factor=6)

    # d_bottleneck4 = inverted_residual_block(d_bottleneck3, 16, strides=(2,2), expansion_factor=6)

    # d_conv1 = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same')(d_bottleneck4)
    # conv3 = Conv2D(128, (1, 1), activation='relu', padding='same')(d_conv1)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)

    model = Model(inputs=[inputs], outputs=[conv2])

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU()])

    return model
    
if __name__ == '__main__':
    model = create_mobile_unet()
    model.summary()
