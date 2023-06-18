import tensorflow as tf
from src.models import Model
from src.layers import DeepInsightEncoding, DomainEncoding, NaiveEncoding, PositionalEncoding, PositionalEncoding2
from src.layers.concrete_dropout import ConcreteSpatialDropout2D, get_weight_regularizer, get_dropout_regularizer


class UNet(Model):
    def __init__(self, name: str, input_dim: int, output_dim, x_train, base_filters: int = 64,
                 activation: str = 'relu', initializer: str = 'he_normal',
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

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(input_layer)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c1)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                          kernel_initializer=self.initializer, padding='same')
        c2 = ConcreteSpatialDropout2D(c2_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                      is_mc_dropout=self.is_mc_dropout)(p1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c2)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

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
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

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
        c3_resized = tf.keras.layers.Resizing(u7.shape[1], u7.shape[2])(c3)
        u7 = tf.keras.layers.concatenate([u7, c3_resized], axis=3)
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
        c2_resized = tf.keras.layers.Resizing(u8.shape[1], u8.shape[2])(c2)
        u8 = tf.keras.layers.concatenate([u8, c2_resized], axis=3)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='valid')(c8)
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
        c9 = tf.keras.layers.BatchNormalization()(c9)

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)


class UNet2(Model):
    def __init__(self, name: str, input_dim: int, output_dim, dropouts, base_filters: int = 64,
                 activation: str = 'relu', initializer: str = 'he_normal',
                 encoding: str = 'naive', positional_encoding: int = 0):
        self.base_filters = base_filters
        self.initializer = initializer
        self.activation = activation
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        self.dropouts = dropouts
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

        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(input_layer)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c1 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c1)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(self.dropouts[0])(p1)

        # c2_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
        #                                  kernel_initializer=self.initializer, padding='same')
        # c2 = ConcreteSpatialDropout2D(c2_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr)(p1)
        c2 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(p1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c2 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c2)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(self.dropouts[1])(p2)

        # c3_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
        #                                  kernel_initializer=self.initializer, padding='same')
        # c3 = ConcreteSpatialDropout2D(c3_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr)(p2)
        c3 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(p2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c3 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c3)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(c3)
        p3 = tf.keras.layers.Dropout(self.dropouts[2])(p3)

        # c4_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
        #                                  kernel_initializer=self.initializer, padding='same')
        # c4 = ConcreteSpatialDropout2D(c4_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr)(p3)
        c4 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(p3)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c4 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c4)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = tf.keras.layers.Dropout(self.dropouts[3])(p4)

        # c5_layer = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3),
        #                                  activation=self.activation,
        #                                  kernel_initializer=self.initializer, padding='same')
        # c5 = ConcreteSpatialDropout2D(c5_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr)(p4)
        c5 = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(p4)
        c5 = tf.keras.layers.BatchNormalization()(c5)
        c5 = tf.keras.layers.Conv2D(filters=self.base_filters * 16, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c5)
        c5 = tf.keras.layers.BatchNormalization()(c5)

        u6 = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 8, kernel_size=(2, 2), strides=(2, 2),
                                             padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4], axis=3)
        c6 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u6)
        c6 = tf.keras.layers.BatchNormalization()(c6)
        c6 = tf.keras.layers.Conv2D(filters=self.base_filters * 8, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c6)
        c6 = tf.keras.layers.BatchNormalization()(c6)

        u7 = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 4, kernel_size=(2, 2), strides=(2, 2),
                                             padding='same')(c6)
        c3_resized = tf.keras.layers.Resizing(u7.shape[1], u7.shape[2])(c3)
        u7 = tf.keras.layers.concatenate([u7, c3_resized], axis=3)
        c7 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u7)
        c7 = tf.keras.layers.BatchNormalization()(c7)
        c7 = tf.keras.layers.Conv2D(filters=self.base_filters * 4, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c7)
        c7 = tf.keras.layers.BatchNormalization()(c7)

        u8 = tf.keras.layers.Conv2DTranspose(filters=self.base_filters * 2, kernel_size=(2, 2), strides=(2, 2),
                                             padding='same')(c7)
        c2_resized = tf.keras.layers.Resizing(u8.shape[1], u8.shape[2])(c2)
        u8 = tf.keras.layers.concatenate([u8, c2_resized], axis=3)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u8)
        c8 = tf.keras.layers.BatchNormalization()(c8)
        c8 = tf.keras.layers.Conv2D(filters=self.base_filters * 2, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='valid')(c8)
        c8 = tf.keras.layers.BatchNormalization()(c8)

        u9 = tf.keras.layers.Conv2DTranspose(filters=self.base_filters, kernel_size=(2, 2), strides=(2, 2),
                                             padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(u9)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        c9 = tf.keras.layers.Conv2D(filters=self.base_filters, kernel_size=(3, 3), activation=self.activation,
                                    kernel_initializer=self.initializer, padding='same')(c9)
        c9 = tf.keras.layers.BatchNormalization()(c9)

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(c9)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)
