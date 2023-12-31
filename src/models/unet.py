import tensorflow as tf
from src.models import Model
from src.layers import DeepInsightEncoding, DomainEncoding, NaiveEncoding, PositionalEncoding, PositionalEncoding2
from src.layers.concrete_dropout import ConcreteSpatialDropout2D, get_weight_regularizer, get_dropout_regularizer


class UNet(Model):
    """
    UNet is an implementation of the U-Net model  with concrete dropout functionality for medical imaging segmentation.

    References:
    - Olaf Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" - arXiv:1505.04597

    Attributes:
    - base_filters (int): The initial number of filters for the convolutional layers.
    - kernel_size (int): Size of the convolutional kernel.
    - initializer (str): Initializer for the weights of layers.
    - activation (str): Activation function used in the network.
    - encoding (str): Type of input data encoding to use; options include 'deepinsight', 'domain', 'domain_lengths', and 'naive'.
    - positional_encoding (int): Type of positional encoding to apply. Default is 0 (no encoding).
    - wr (float): Weight regularization parameter.
    - dr (float): Dropout regularization parameter.
    - is_mc_dropout (bool): Dropout regularization parameter.
    - x_train (ndarray): Training dataset, used if encoding is 'deepinsight'.
    """

    def __init__(self, name: str, input_dim: int, output_dim, train_size: int = 100, base_filters: int = 64,
                 activation: str = 'relu', initializer: str = 'he_normal', x_train=None,
                 encoding: str = 'naive', positional_encoding: int = 0, is_mc_dropout: bool = False) -> None:
        self.base_filters = base_filters
        self.initializer = initializer
        self.activation = activation
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        self.wr = get_weight_regularizer(train_size, l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(train_size, tau=1.0)
        self.is_mc_dropout = is_mc_dropout
        self.x_train = x_train
        super().__init__(name, input_dim, output_dim)

    def build(self) -> None:
        input_tensor = tf.keras.layers.Input(self.input_dim)

        if self.encoding == 'deepinsight' and self.x_train is not None:
            custom_input_layer = DeepInsightEncoding(x_train=self.x_train)
        elif self.encoding == 'domain':
            custom_input_layer = DomainEncoding()
        elif self.encoding == 'domain_lengths':
            custom_input_layer = DomainEncoding(use_lengths=True)
        elif self.encoding == 'naive':
            custom_input_layer = NaiveEncoding()
        else:
            raise ValueError(f'Unknown encoding type: {self.encoding}')
        input_layer = custom_input_layer(input_tensor)

        if self.positional_encoding == 1:
            input_layer = PositionalEncoding()(input_layer)
        elif self.positional_encoding == 2:
            input_layer = PositionalEncoding2()(input_layer)

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(input_layer)
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

        c5_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3),
                                          activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c5 = ConcreteSpatialDropout2D(c5_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p4)
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
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)
