'''
poolNet in paper <<A Simple Pooling-Based Design for Real-Time Salient Object Detection>>
https://arxiv.org/abs/1904.09569
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


def feature_aggregation_module(x):
    pp1 = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    pp1 = conv_block(pp1, x._keras_shape[-1], (3, 3))
    pp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(pp1)
    pp2 = AveragePooling2D(pool_size=(4, 4), padding='same')(x)
    pp2 = conv_block(pp2, x._keras_shape[-1], (3, 3))
    pp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(pp2)
    pp3 = AveragePooling2D(pool_size=(8, 8), padding='same')(x)
    pp3 = conv_block(pp3, x._keras_shape[-1], (3, 3))
    pp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(pp3)
    all = Add()([pp1, pp2, pp3, x])
    return conv_block(all, x._keras_shape[-1], (3, 3))


def VGG16_POOLNet(input_shape, weights=None, classes=1):
   
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

    center = PyramidPoolingModule()(m1_down16)

    a1 = feature_aggregation_module(m1_down16)

    mm_down8 = Add()(
        [decoder_block(center, 512, strides=(2, 2)), decoder_block(a1, 512, strides=(2, 2)), m1_down8])  # 8
    mm_down8 = conv_block(mm_down8, 256, (3, 3))
    mm_down8 = conv_block(mm_down8, 256, (3, 3))

    a2 = feature_aggregation_module(mm_down8)
   

    mm_down4 = Add()(
        [m1_down4, decoder_block(a2, 256, strides=(2, 2)), decoder_block(center, 256, strides=(4, 4))])  # 4
    mm_down4 = conv_block(mm_down4, 128, (3, 3))
   
    a3 = feature_aggregation_module(mm_down4)

    mm_down2 = Add()(
        [m1_down2, decoder_block(a3, 128, strides=(2, 2)), decoder_block(center, 128, strides=(8, 8))])  # 2
    mm_down2 = conv_block(mm_down2, 64, (3, 3))
    
    a4 = feature_aggregation_module(mm_down2)
    mm_down = decoder_block(a4, 64, strides=(2, 2))
    mm_down = conv_block(mm_down, 64, (3, 3))
   

    promap = Conv2D(classes, (1, 1), activation='sigmoid')(mm_down)
    model = Model(inputs=[base_model1.input], outputs=[promap])
    return model