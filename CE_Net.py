'''
CE-Net in paper <<CE-Net: Context Encoder Network for 2D Medical Image Segmentation>>
https://arxiv.org/abs/1903.02740
back-bone:VGG16
'''
from keras.applications.vgg16 import VGG16

from keras.layers import Conv2D, UpSampling2D, concatenate, BatchNormalization, MaxPooling2D, Activation, \
    Conv2DTranspose, AveragePooling2D, Add
import keras.backend as K
from keras.models import Model

from keras_pyramid_pooling_module import PyramidPoolingModule

channel_axis = 1 if K.image_data_format() == "channels_first" else -1


def conv_block(input, num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    '''
    most of the convolution now consist of convolution layer, Relu activation layer and BatchNormalizarion layer
    Here I put them together for convenience
    :param input:
    :param num_filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :return:
    '''
    conv1 = Conv2D(num_filters, kernel_size, strides=strides, padding=padding, activation='relu')(input)
    conv1 = BatchNormalization()(conv1)
    return conv1


def decoder_block(x, num_filters, strides):
    out_filters = x._keras_shape[3] // 4

    conv1 = conv_block(x, out_filters, (1, 1))
    conv2 = Conv2DTranspose(out_filters, (3, 3), strides=strides, padding='same', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = conv_block(conv2, num_filters, (1, 1))
    return conv3


def ce_block(x):
    '''
    context entractor module
    :param x:input
    :return:
    '''
    dilated_1 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=(1, 1))(x)
    dilated_2 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=(2, 2))(dilated_1)
    dilated_3 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=(4, 4))(dilated_2)
    dilated_4 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=(8, 8))(dilated_3)

    center = concatenate([x, dilated_1, dilated_2, dilated_3, dilated_4], axis=channel_axis)
    return center


def VGG16_CE_Unet(input_shape, weights=None, classes=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    base_model1 = VGG16(include_top=False,
                        weights=weights,
                        input_tensor=None,
                        input_shape=input_shape,
                        pooling=None,
                        classes=classes)
    # base_model1.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in base_model1.layers:
        layer.trainable = True

    m1_down2 = base_model1.get_layer('block2_conv2').output  # 2
    m1_down4 = base_model1.get_layer('block3_conv3').output  # 4
    m1_down8 = base_model1.get_layer('block4_conv3').output  # 8
    m1_down16 = base_model1.get_layer('block5_conv3').output  # 16

    # CE-block

    center = ce_block(m1_down16)
	
    center = conv_block(center, 512, (3, 3))
	
    center = PyramidPoolingModule()(center)
	
    mm_down8 = concatenate([decoder_block(center, 512), m1_down8], axis=channel_axis)  # 8
    mm_down8 = conv_block(mm_down8, 256, (3, 3))
    mm_down8 = conv_block(mm_down8, 256, (3, 3))

    

    mm_down4 = concatenate([m1_down4, decoder_block(mm_down8, 256)], axis=channel_axis)  # 4
    mm_down4 = conv_block(mm_down4, 128, (3, 3))
    mm_down4 = conv_block(mm_down4, 128, (3, 3))
    

    mm_down2 = concatenate([m1_down2, decoder_block(mm_down4, 128)], axis=channel_axis)  # 2
    mm_down2 = conv_block(mm_down2, 64, (3, 3))
    mm_down2 = conv_block(mm_down2, 64, (3, 3))
   
    mm_down = decoder_block(mm_down2, 64)
    mm_down = conv_block(mm_down, 32, (3, 3))
    mm_down = conv_block(mm_down, 32, (3, 3))
   

    promap = Conv2D(classes, (1, 1), activation='sigmoid')(mm_down)
    model = Model(inputs=[base_model1.input], outputs=[promap])
    return model
