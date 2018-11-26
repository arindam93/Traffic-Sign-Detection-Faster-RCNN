from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    return ''


def img_length_calc_function(C, width, height):
    def get_output_length(input_length):
        return input_length / C.rpn_stride

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):
    if input_tensor is None:
        img_input = Input(shape=(None, None, 3))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=(None, None, 3))
        else:
            img_input = input_tensor

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu')(img_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(pool2)
    x = pool2
    return x


def rpn(base_layers, num_anchors):
    """
    The RPN network that takes feature map as input and return region proposals with probability
    of having an object (classification) and bbox (regression)

    :param base_layers:  feature map from base ConvNet
    """
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=44, trainable=False):
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



