import tensorflow as tf
from src.models import Model
from src.layers import PositionalEncoding, PositionalEncoding2
import math
from src.layers.concrete_dropout import ConcreteDenseDropout, ConcreteSpatialDropout2D, get_weight_regularizer, \
    get_dropout_regularizer


class MultiPathUNet(Model):
    """
    MultiPathUNet is a multi-path implementation of the U-Net model  with concrete dropout functionality for medical imaging segmentation.

    References:
    - Olaf Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" - arXiv:1505.04597

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

    def build(self) -> None:
        input_tensor = tf.keras.layers.Input(self.input_matrix_dim)
        c0 = input_tensor

        if self.positional_encoding == 1:
            c0 = PositionalEncoding()(c0)
        elif self.positional_encoding == 2:
            c0 = PositionalEncoding2()(c0)

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c0)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c1)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c1)

        c2_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c2 = ConcreteSpatialDropout2D(c2_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c2)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c2)

        c3_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c3 = ConcreteSpatialDropout2D(c3_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c3)

        c4_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c4 = ConcreteSpatialDropout2D(c4_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p3)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c4)

        vector_input = tf.keras.layers.Input(self.input_vector_dim)
        v = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(vector_input)
        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        v = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(v)
        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        v = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(v)
        z = round(math.prod(self.input_matrix_dim) / (p4.shape[1] * p4.shape[2]))
        v = tf.keras.layers.Dense(p4.shape[1] * p4.shape[2] * z, activation=self.activation)(v)
        v_reshaped = tf.keras.layers.Reshape((p4.shape[1], p4.shape[2], z))(v)
        combined = tf.keras.layers.concatenate([p4, v_reshaped], axis=3)
        combined = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')(combined)
        combined = tf.keras.layers.BatchNormalization()(combined)

        c5_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3),
                                          activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c5 = ConcreteSpatialDropout2D(c5_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(combined)
        c5 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)

        u6_layer = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 8, kernel_size=(2, 2), strides=(2, 2),
                                                   padding='same')
        u6 = ConcreteSpatialDropout2D(u6_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(c5)
        u6 = tf.keras.layers.concatenate([u6, c4], axis=3)
        c6 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
        c6 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)

        u7_layer = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 4, kernel_size=(2, 2), strides=(2, 2),
                                                   padding='same')
        u7 = ConcreteSpatialDropout2D(u7_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(c6)
        u7 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(2, 2), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='valid')(u7)
        u7 = tf.keras.layers.BatchNormalization()(u7)
        u7 = tf.keras.layers.concatenate([u7, c3], axis=3)
        c7 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
        c7 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)

        u8_layer = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 2, kernel_size=(2, 2), strides=(2, 2),
                                                   padding='same')
        u8 = ConcreteSpatialDropout2D(u8_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(c7)
        u8 = tf.keras.layers.concatenate([u8, c2], axis=3)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)

        u9_layer = tf.keras.layers.Conv2DTranspose(filters=self.base_filters, kernel_size=(2, 2), strides=(2, 2),
                                                   padding='same')
        u9 = ConcreteSpatialDropout2D(u9_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c9)

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

        self.model = tf.keras.models.Model(inputs=[input_tensor, vector_input], outputs=output_layer)
