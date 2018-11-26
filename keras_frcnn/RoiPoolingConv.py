from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    """
    ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size(int): Size of pooling region to use. pool_size = 7 will result in a 7x7 region
        num_rois: number of RoI to be used
    # Input shape
        list of two 4D tensors [X_img, X_roi] with shape:
        X_img: (1, rows, cols, channels)
        X_roi: (1,num_rois,4) list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape: (1, num_rois, channels, pool_size, pool_size)
    """

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering == 'tf', 'dim_ordering must be tf'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]
        """
        assert self.dim_ordering == 'tf', 'dim_ordering must be tf'
        self.nb_channels = input_shape[0][3]


    def compute_output_shape(self, input_shape):
        """
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels
        """
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert(len(x) == 2)
        img, rois = x[0], x[1]
        input_shape = K.shape(img)
        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            assert self.dim_ordering == 'tf', 'dim_ordering must be tf'
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
