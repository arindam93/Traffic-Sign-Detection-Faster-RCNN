#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
    import tensorflow as tf

# """
# Four Losses: 
# 1. rpn_loss_cls: RPN classify loss: object / not object 
# 2. rpn_loss_regr: RPN regress box coordinates 
# 3. class_loss_cls: Final classification score (object classes)
# 4. class_loss_regr: Final box coordinates 
# For RPN losses: 
# L({pi}, {ti}) = Loss_cls + Loss_reg = SUM(L_cls(pi, pi*)) / N_cls + lambda * SUM(L_reg(ti, ti*)) / N_reg, 
# L_cls: log loss over two classes(obj / not obj)
# L_reg: robust loss (smooth L1 in this case), L_reg(ti, ti*) = R(ti − t*)
# pi: predicted probability of anchor i being an object
# pi_star: 1 for pos anchor and 0 for neg anchor
# N_cls/N_reg: number of anchors
# ground-truth label p*: 1 if the anchor i is positive, 0 if the anchor is negative
# ti: 4 parameterized coordinates of the predicted bounding box
# t*: 4 parameterized coordinates of the i ground-truth box associated with a positive anchor
# """



def rpn_loss_cls(num_anchors):
    """
    y_true = [y_is_box_valid, y_rpn_overlap]
    :param num_anchors:
    :return:
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        N_cls = K.sum(1e-5 + y_true[:, :, :, :num_anchors])
        total_cls_loss = K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_true[:, :, :, num_anchors:], y_pred[:, :, :, :]))
        rpn_cls_loss = total_cls_loss / N_cls
        return rpn_cls_loss

    return rpn_loss_cls_fixed_num


def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        d = y_true[:, :, :, 4 * num_anchors:] - y_pred
        p_star = K.cast(K.less_equal(K.abs(d), 1.0), tf.float32)
        N_reg = K.sum(1e-5 + y_true[:, :, :, :4*num_anchors])

        l1_smooth = 0.5 * d * d if p_star == 1 else K.abs(d) - 0.5
        total_reg_loss = K.sum(y_true[:, :, :, :4*num_anchors] * l1_smooth)
        rpn_reg_loss = total_reg_loss / N_reg
        return rpn_reg_loss

    return rpn_loss_regr_fixed_num


"""
For classification losses: 
Regression Loss: the bounding coordinates 
Classification Loss: final cross-entropy score 
"""

def class_loss_cls(y_true, y_pred):
    class_cls_loss = K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
    return class_cls_loss


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        d = y_true[:, :, 4*num_classes:] - y_pred
        p_star = K.cast(K.less_equal(K.abs(d), 1.0), tf.float32)
        N_reg = K.sum(1e-5 + y_true[:, :, :4*num_classes])

        l1_smooth = 0.5 * d * d if p_star == 1 else K.abs(d) - 0.5
        total_reg_loss = K.sum(y_true[:, :, :4*num_classes] * l1_smooth)
        class_reg_loss = total_reg_loss / N_reg
        return class_reg_loss

    return class_loss_regr_fixed_num
