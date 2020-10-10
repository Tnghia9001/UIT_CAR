"""Lite R-ASPP Semantic Segmentation based on MobileNetV3.
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Multiply, Add, Reshape, Lambda, ReLU, Dropout, Flatten, Dense
from keras.utils.vis_utils import plot_model
from model.layers.bilinear_upsampling import BilinearUpSampling2D
# from tensorflow.image import ResizeMethod

class LiteRASSP:
    def __init__(self, input_shape = (512, 512, 3), n_class=1, alpha=1.0, weights=None, backbone='small'):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor (should be 1024 × 2048 or 512 × 1024 according 
                to the paper).
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier for mobilenetV3.
            weights: String, weights for mobilenetv3.
            backbone: String, name of backbone (must be small or large).
        """
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights
        self.backbone = backbone

    def build(self, plot=False):
        """build Lite R-ASPP.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        from model.mobilenet_v3_small import MobileNetV3_Small
        model = MobileNetV3_Small(self.shape, 1, alpha=1.0, include_top=False).build()
        inputs = model.input

        # x = model.get_layer('activation_25').output
        x = model.get_layer('global_average_pooling2d_10').output
        x = Dropout(0.5)(x)
        # x = Flatten()(x)
        x = Dense(100, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Dense(50, activation='elu')(x)
        x = Dense(10, activation='elu')(x)
        x = Dense(1)(x)

        model = Model(inputs=inputs, outputs=x)

        if plot:
            plot_model(model, to_file='images/LR_ASPP.png', show_shapes=True)

        return model





