from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout, Add, DepthwiseConv2D, Activation
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryIoU
from keras.applications import MobileNetV2

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

def create_mobile_unet(image_size=(128,128,3), pretrained=True):
    """returns unet model"""

    inputs = Input(image_size)
    
    #Encoder
    if pretrained:
        encoder = MobileNetV2(include_top=False, input_tensor=inputs, alpha=0.5)
        s1 = inputs               
        s2 = encoder.get_layer("block_1_expand_relu").output    
        s3 = encoder.get_layer("block_3_expand_relu").output    
        s4 = encoder.get_layer("block_6_expand_relu").output    

        out = encoder.get_layer("block_13_expand_relu").output   

    else : 
        s1 = inputs
        conv1 = Conv2D(32, (3, 3), activation='relu6', padding='same', strides=(2,2))(s1)
        conv1 = BatchNormalization()(conv1)

        s2 = inverted_residual_block(conv1, 16, strides=(1,1), expansion_factor=1)

        s3 = inverted_residual_block(s2, 24, strides=(2,2), expansion_factor=6, n=2)

        s4 = inverted_residual_block(s3, 32, strides=(2,2), expansion_factor=6, n=3)

        bottleneck = inverted_residual_block(s4, 64, strides=(2,2), expansion_factor=6, n=4)

        out = inverted_residual_block(bottleneck, 96, strides=(1,1), expansion_factor=6, n =3)

    #Upsampling
    dconv1 = deconv_block(out, 256, s4)
    dconv2 = deconv_block(dconv1, 128 , s3)
    dconv3 = deconv_block(dconv2, 64, s2)
    dconv4 = deconv_block(dconv3, 32, s1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(dconv4)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['acc', BinaryIoU()])

    return model
    

if __name__ == '__main__':
    model = create_mobile_unet(pretrained=False)
    model.summary()
