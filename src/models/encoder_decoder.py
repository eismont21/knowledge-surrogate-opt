import tensorflow as tf
from src.models import Model
from src.layers import DeepInsightEncoding, DomainEncoding, NaiveEncoding, PositionalEncoding, PositionalEncoding2
from src.layers.concrete_dropout import ConcreteSpatialDropout2D, get_weight_regularizer, get_dropout_regularizer


class EncoderDecoderDropout(Model):
    def __init__(self, name: str, input_dim: int, output_dim, x_train, base_filters: int = 64,
                 activation: str = 'relu', initializer=tf.keras.initializers.GlorotUniform(),
                 encoding: str = 'naive', positional_encoding: int = 0, is_mc_dropout: bool = False):
        self.base_filters = base_filters
        self.initializer = initializer
        self.activation = activation
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        self.x_train = x_train
        self.wr = get_weight_regularizer(self.x_train.shape[0], l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(self.x_train.shape[0], tau=1.0)
        self.is_mc_dropout = is_mc_dropout
        super().__init__(name, input_dim, output_dim)

    def build(self):
        input_tensor = tf.keras.layers.Input(self.input_dim)

        if self.encoding == 'deepinsight':
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

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(15, 15), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(input_layer)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(7, 7), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c2 = ConcreteSpatialDropout2D(c2_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p1)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(5, 5), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c3 = ConcreteSpatialDropout2D(c3_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p2)
        p3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c3)

        c4_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c4 = ConcreteSpatialDropout2D(c4_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p3)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        c5_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c5 = ConcreteSpatialDropout2D(c5_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p4)
        p5 = tf.keras.layers.MaxPooling2D((2, 2))(c5)

        c6_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(2, 2), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c6 = ConcreteSpatialDropout2D(c6_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p5)
        p6 = tf.keras.layers.MaxPooling2D((2, 2))(c6)

        u1 = tf.keras.layers.UpSampling2D((2, 2))(p6)
        u1_resized = tf.keras.layers.Resizing(c6.shape[1], c6.shape[2])(u1)
        c7_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(2, 2), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c7 = ConcreteSpatialDropout2D(c7_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(u1_resized)

        u2 = tf.keras.layers.UpSampling2D((2, 2))(c7)
        u2_resized = tf.keras.layers.Resizing(c5.shape[1], c5.shape[2])(u2)
        c8_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c8 = ConcreteSpatialDropout2D(c8_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(u2_resized)

        u3 = tf.keras.layers.UpSampling2D((2, 2))(c8)
        c9_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                          activation=self.activation, kernel_initializer=self.initializer,
                                          padding='same')
        c9 = ConcreteSpatialDropout2D(c9_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(u3)

        u4 = tf.keras.layers.UpSampling2D((2, 2))(c9)
        c10_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(5, 5), strides=(1, 1),
                                           activation=self.activation, kernel_initializer=self.initializer,
                                           padding='valid')
        c10 = ConcreteSpatialDropout2D(c10_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                       is_mc_dropout=self.is_mc_dropout)(u4)
        c10_resized = tf.keras.layers.Resizing(c3.shape[1], c3.shape[2])(c10)

        u5 = tf.keras.layers.UpSampling2D((2, 2))(c10_resized)
        c11_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(7, 7), strides=(1, 1),
                                           activation=self.activation, kernel_initializer=self.initializer,
                                           padding='same')
        c11 = ConcreteSpatialDropout2D(c11_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                       is_mc_dropout=self.is_mc_dropout)(u5)

        u6 = tf.keras.layers.UpSampling2D((2, 2))(c11)
        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(15, 15), strides=(1, 1),
                                              activation='sigmoid', kernel_initializer=self.initializer,
                                              padding='same')(u6)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)


class EncoderDecoder(Model):
    def __init__(self, name: str, input_dim: int, output_dim, x_train, base_filters: int = 64,
                 activation: str = 'relu', initializer=tf.keras.initializers.GlorotUniform(),
                 encoding: str = 'naive', positional_encoding: int = 0):
        self.x_train = x_train
        self.base_filters = base_filters
        self.initializer = initializer
        self.activation = activation
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        super().__init__(name, input_dim, output_dim)

    def build(self):
        input_tensor = tf.keras.layers.Input(self.input_dim)

        if self.encoding == 'deepinsight':
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

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(15, 15), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(input_layer)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(7, 7), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(p1)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(5, 5), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(p2)
        p3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c3)

        c4 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(p3)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(p4)
        p5 = tf.keras.layers.MaxPooling2D((2, 2))(c5)

        c6 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(2, 2), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(p5)
        p6 = tf.keras.layers.MaxPooling2D((2, 2))(c6)

        u1 = tf.keras.layers.UpSampling2D((2, 2))(p6)
        u1_resized = tf.keras.layers.Resizing(c6.shape[1], c6.shape[2])(u1)
        c7 = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(2, 2), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(u1_resized)

        u2 = tf.keras.layers.UpSampling2D((2, 2))(c7)
        u2_resized = tf.keras.layers.Resizing(c5.shape[1], c5.shape[2])(u2)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(u2_resized)

        u3 = tf.keras.layers.UpSampling2D((2, 2))(c8)
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                    activation=self.activation, kernel_initializer=self.initializer,
                                    padding='same')(u3)

        u4 = tf.keras.layers.UpSampling2D((2, 2))(c9)
        c10 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(5, 5), strides=(1, 1),
                                     activation=self.activation, kernel_initializer=self.initializer,
                                     padding='valid')(u4)
        c10_resized = tf.keras.layers.Resizing(c3.shape[1], c3.shape[2])(c10)

        u5 = tf.keras.layers.UpSampling2D((2, 2))(c10_resized)
        c11 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(7, 7), strides=(1, 1),
                                     activation=self.activation, kernel_initializer=self.initializer,
                                     padding='same')(u5)

        u6 = tf.keras.layers.UpSampling2D((2, 2))(c11)
        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(15, 15), strides=(1, 1),
                                              activation='sigmoid', kernel_initializer=self.initializer,
                                              padding='same')(u6)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)
