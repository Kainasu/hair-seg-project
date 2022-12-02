from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout, Add, DepthwiseConv2D, Activation
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU

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
    
    #Contraction
    
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
    
    #Expansion

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

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc'])

    return model
    
if __name__ == '__main__':
    model = create_mobile_unet()
    model.summary()
