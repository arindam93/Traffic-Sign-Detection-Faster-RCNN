from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    '''
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
    '''
    return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def img_length_calc_function(C, width, height):
    def get_output_length(input_length):
        return input_length / C.rpn_stride
    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):
    """
    Based ConvNet shared by both RPN and ROI Pooling layer, implemented by a midified VGG-16,
    and returns a feature map.
    C.rpn_stride is set such that it corresponds to this base network
    """

    if input_tensor is None:
        img_input = Input(shape=(None, None, 3))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=(None, None, 3))
        else:
            img_input = input_tensor

    bn_axis = 3

    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv2)

    # Block 2
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv3)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv4)

    # Block 3
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv5)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv6)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv7)

    # Block 4
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv8)
    conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv9)
    #pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv10)

    # Block 5
    conv11 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(conv10)
    conv12 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv11)
    conv13 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv12)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = conv13
    return x

def rpn(base_layers, num_anchors):
    """
    The RPN network that takes feature map as input and return region proposals with probability
    of having an object (classification) and bbox (regression)

    :param base_layers:  feature map from base ConvNet
    """
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):
    """
    The classifier network that takes feature map as input and apply RoI pooling

    :param base_layers: feature map from base ConvNet
    :param input_rois: RoIs prposed by RPN
    :param num_rois: number of RoIs at one time
    """
    pooling_regions = 7
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    #out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    #out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


