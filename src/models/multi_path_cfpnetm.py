import tensorflow as tf
from src.models import Model
from src.layers import PositionalEncoding, PositionalEncoding2
import math
from src.layers.concrete_dropout import ConcreteDenseDropout, ConcreteSpatialDropout2D, get_weight_regularizer, \
    get_dropout_regularizer
from keras.layers import add


class MultiPathCFPNetM(Model):
    """
    MultiPathCFPNetM is a multi-path implementation of the CFPNet-M model with concrete dropout functionality for medical imaging segmentation.

    References:
    - Ange Lou et al. "CFPNet-M: A Light-Weight Encoder-Decoder Based Network for Multimodal Biomedical Image Real-Time Segmentation" - arXiv:2103.12212

    Attributes:
    - input_matrix_dim: Dimensionality of the matrix input.
    - input_vector_dim: Dimensionality of the vector input.
    - base_filters (int): The initial number of filters for the convolutional layers.
    - initializer (str): Initializer for the weights of layers.
    - activation (str): Activation function used in the network.
    - hidden_neurons (int): Number of neurons in the hidden layers.
    - positional_encoding (int): Type of positional encoding to apply. Default is 0 (no encoding).
    - wr (float): Weight regularization parameter.
    - dr (float): Dropout regularization parameter.
    - is_mc_dropout (bool): Dropout regularization parameter.
    """

    def __init__(self, name: str, input_dim, output_dim, train_size: int = 100, base_filters: int = 64,
                 activation: str = 'relu', initializer: str = 'he_normal', hidden_neurons: int = 500,
                 positional_encoding: int = 0, is_mc_dropout: bool = False) -> None:
        self.input_matrix_dim, self.input_vector_dim = input_dim
        self.base_filters = base_filters
        self.initializer = initializer
        self.activation = activation
        self.hidden_neurons = hidden_neurons
        self.positional_encoding = positional_encoding
        self.wr = get_weight_regularizer(train_size, l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(train_size, tau=1.0)
        self.is_mc_dropout = is_mc_dropout
        super().__init__(name, input_dim, output_dim)

    def conv2d_bn(self, x: tf.Tensor, filters: int, kernel_size=None, d_rate: int = 1, strides: (int, int) = (1, 1),
                  padding: str = 'same', activation: str = 'relu', groups: int = 1) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=d_rate,
                                   strides=strides, activation=activation, padding=padding,
                                   kernel_initializer=self.initializer, groups=groups)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return x

    def cfp_module(self, inp: tf.Tensor, filters: int, d_size: int) -> tf.Tensor:
        x_inp = self.conv2d_bn(inp, filters // 4, kernel_size=(1, 1))
        x_1_1 = self.conv2d_bn(x_inp, filters // 16, groups=filters // 16)
        x_1_2 = self.conv2d_bn(x_1_1, filters // 16, groups=filters // 16)
        x_1_3 = self.conv2d_bn(x_1_2, filters // 8, groups=filters // 16)

        x_2_1 = self.conv2d_bn(x_inp, filters // 16, d_rate=d_size // 4 + 1, groups=filters // 16)
        x_2_2 = self.conv2d_bn(x_2_1, filters // 16, d_rate=d_size // 4 + 1, groups=filters // 16)
        x_2_3 = self.conv2d_bn(x_2_2, filters // 8, d_rate=d_size // 4 + 1, groups=filters // 16)

        x_3_1 = self.conv2d_bn(x_inp, filters // 16, d_rate=d_size // 2 + 1, groups=filters // 16)
        x_3_2 = self.conv2d_bn(x_3_1, filters // 16, d_rate=d_size // 2 + 1, groups=filters // 16)
        x_3_3 = self.conv2d_bn(x_3_2, filters // 8, d_rate=d_size // 2 + 1, groups=filters // 16)

        x_4_1 = self.conv2d_bn(x_inp, filters // 16, d_rate=d_size + 1, groups=filters // 16)
        x_4_2 = self.conv2d_bn(x_4_1, filters // 16, d_rate=d_size + 1, groups=filters // 16)
        x_4_3 = self.conv2d_bn(x_4_2, filters // 8, d_rate=d_size + 1, groups=filters // 16)

        o_1 = tf.keras.layers.concatenate([x_1_1, x_1_2, x_1_3], axis=3)
        o_2 = tf.keras.layers.concatenate([x_2_1, x_2_2, x_2_3], axis=3)
        o_3 = tf.keras.layers.concatenate([x_1_1, x_3_2, x_3_3], axis=3)
        o_4 = tf.keras.layers.concatenate([x_1_1, x_4_2, x_4_3], axis=3)

        o_1 = tf.keras.layers.BatchNormalization()(o_1)
        o_2 = tf.keras.layers.BatchNormalization()(o_2)
        o_3 = tf.keras.layers.BatchNormalization()(o_3)
        o_4 = tf.keras.layers.BatchNormalization()(o_4)

        ad1 = o_1
        ad2 = add([ad1, o_2])
        ad3 = add([ad2, o_3])
        ad4 = add([ad3, o_4])
        output = tf.keras.layers.concatenate([ad1, ad2, ad3, ad4], axis=3)
        output = tf.keras.layers.BatchNormalization()(output)
        output = self.conv2d_bn(output, filters, kernel_size=(1, 1), padding='valid')
        output = add([output, inp])

        return output

    def build(self) -> None:
        input_tensor = tf.keras.layers.Input(self.input_matrix_dim)
        c0 = input_tensor

        if self.positional_encoding == 1:
            c0 = PositionalEncoding()(c0)
        elif self.positional_encoding == 2:
            c0 = PositionalEncoding2()(c0)

        conv1 = self.conv2d_bn(c0, filters=self.base_filters, strides=(2, 2))
        conv2 = self.conv2d_bn(conv1, filters=self.base_filters)
        conv3 = self.conv2d_bn(conv2, filters=self.base_filters)

        injection_1 = tf.keras.layers.AveragePooling2D(padding='same')(c0)
        conv_layer = tf.keras.layers.Conv2D(filters=injection_1.shape[-1], kernel_size=(1, 1), padding='same',
                                            activation=self.activation, kernel_initializer=self.initializer)
        injection_1 = ConcreteSpatialDropout2D(conv_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                               is_mc_dropout=self.is_mc_dropout)(injection_1)
        injection_1 = tf.keras.layers.BatchNormalization()(injection_1)
        opt_cat_1 = tf.keras.layers.concatenate([conv3, injection_1], axis=3)

        # CFP block 1
        opt_cat_1_0 = self.conv2d_bn(opt_cat_1, filters=self.base_filters * 2, strides=(2, 2))
        cfp_1 = self.cfp_module(opt_cat_1_0, self.base_filters * 2, 2)
        cfp_2 = self.cfp_module(cfp_1, self.base_filters * 2, 2)

        injection_2 = tf.keras.layers.AveragePooling2D(padding='same')(injection_1)
        conv_layer = tf.keras.layers.Conv2D(filters=injection_2.shape[-1], kernel_size=(1, 1), padding='same',
                                            activation=self.activation, kernel_initializer=self.initializer)
        injection_2 = ConcreteSpatialDropout2D(conv_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                               is_mc_dropout=self.is_mc_dropout)(injection_2)
        injection_2 = tf.keras.layers.BatchNormalization()(injection_2)
        opt_cat_2 = tf.keras.layers.concatenate([cfp_2, opt_cat_1_0, injection_2], axis=3)

        # CFP block 2
        opt_cat_2_0 = self.conv2d_bn(opt_cat_2, filters=self.base_filters * 4, strides=(2, 2))
        cfp_3 = self.cfp_module(opt_cat_2_0, self.base_filters * 4, 4)
        cfp_4 = self.cfp_module(cfp_3, self.base_filters * 4, 4)
        cfp_5 = self.cfp_module(cfp_4, self.base_filters * 4, 8)
        cfp_6 = self.cfp_module(cfp_5, self.base_filters * 4, 8)
        cfp_7 = self.cfp_module(cfp_6, self.base_filters * 4, 16)
        cfp_8 = self.cfp_module(cfp_7, self.base_filters * 4, 16)

        injection_3 = tf.keras.layers.AveragePooling2D(padding='same')(injection_2)
        conv_layer = tf.keras.layers.Conv2D(filters=injection_3.shape[-1], kernel_size=(1, 1), padding='same',
                                            activation=self.activation, kernel_initializer=self.initializer)
        injection_3 = ConcreteSpatialDropout2D(conv_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                               is_mc_dropout=self.is_mc_dropout)(injection_3)
        injection_3 = tf.keras.layers.BatchNormalization()(injection_3)
        opt_cat_3 = tf.keras.layers.concatenate([cfp_8, opt_cat_2_0, injection_3], axis=3)

        vector_input = tf.keras.layers.Input(self.input_vector_dim)
        v = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(vector_input)
        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        v = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(v)
        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        v = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(v)
        z = round(math.prod(self.input_matrix_dim) / (opt_cat_3.shape[1] * opt_cat_3.shape[2]))
        v = tf.keras.layers.Dense(opt_cat_3.shape[1] * opt_cat_3.shape[2] * z, activation=self.activation)(v)
        v_reshaped = tf.keras.layers.Reshape((opt_cat_3.shape[1], opt_cat_3.shape[2], z))(v)
        combined = tf.keras.layers.concatenate([opt_cat_3, v_reshaped], axis=3)
        combined = tf.keras.layers.Conv2D(filters=opt_cat_3.shape[-1], kernel_size=(3, 3),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')(combined)
        combined = tf.keras.layers.BatchNormalization()(combined)

        conv4_layer = tf.keras.layers.Conv2DTranspose(self.base_filters * 4, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same')
        conv4 = ConcreteSpatialDropout2D(conv4_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                         is_mc_dropout=self.is_mc_dropout)(combined)
        conv4 = self.conv2d_bn(conv4, filters=self.base_filters * 4, kernel_size=(2, 2), padding='valid')
        up_1 = tf.keras.layers.concatenate([conv4, opt_cat_2], axis=3)

        conv5_layer = tf.keras.layers.Conv2DTranspose(self.base_filters * 2, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same')
        conv5 = ConcreteSpatialDropout2D(conv5_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                         is_mc_dropout=self.is_mc_dropout)(up_1)
        up_2 = tf.keras.layers.concatenate([conv5, opt_cat_1], axis=3)

        conv6_layer = tf.keras.layers.Conv2DTranspose(self.base_filters, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same')
        conv6 = ConcreteSpatialDropout2D(conv6_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                         is_mc_dropout=self.is_mc_dropout)(up_2)

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv6)

        self.model = tf.keras.models.Model(inputs=[input_tensor, vector_input], outputs=output_layer)
