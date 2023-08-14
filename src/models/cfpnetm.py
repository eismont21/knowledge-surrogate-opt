import tensorflow as tf
from src.models import Model
from src.layers import DeepInsightEncoding, DomainEncoding, NaiveEncoding, PositionalEncoding, PositionalEncoding2
from src.layers.concrete_dropout import ConcreteSpatialDropout2D, get_weight_regularizer, get_dropout_regularizer
from keras.layers import add


class CFPNetM(Model):
    '''
    Source: https://github.com/AngeLouCN/CFPNet-Medicine/tree/main
    '''

    def __init__(self, name: str, input_dim: int, output_dim, train_size: int = 100, base_filters: int = 64,
                 activation: str = 'relu', initializer: str = 'he_normal', x_train=None,
                 encoding: str = 'naive', positional_encoding: int = 0, is_mc_dropout: bool = False):
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

    def set_train_size(self, train_size: int):
        self.wr = get_weight_regularizer(train_size, l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(train_size, tau=1.0)
        for layer in self.model.layers:
            if isinstance(layer, ConcreteSpatialDropout2D):
                layer.set_regularizers(self.wr, self.dr)

    def conv2d_bn(self, x, filters, kernel_size=(3, 3), d_rate=1, strides=(1, 1), padding='same', activation='relu',
                  groups=1):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=d_rate,
                                   strides=strides, activation=activation, padding=padding,
                                   kernel_initializer=self.initializer, groups=groups)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return x

    def cfp_module(self, inp, filters, d_size):
        '''
        CFP module for medicine

        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        '''
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

    def build(self):
        input_tensor = tf.keras.layers.Input(self.input_dim)

        # Encoding Layer
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

        # Positional Encoding
        if self.positional_encoding == 1:
            input_layer = PositionalEncoding()(input_layer)
        elif self.positional_encoding == 2:
            input_layer = PositionalEncoding2()(input_layer)

        conv1 = self.conv2d_bn(input_layer, filters=self.base_filters, strides=(2, 2))
        conv2 = self.conv2d_bn(conv1, filters=self.base_filters)
        conv3 = self.conv2d_bn(conv2, filters=self.base_filters)

        injection_1 = tf.keras.layers.AveragePooling2D(padding='same')(input_layer)
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

        conv4_layer = tf.keras.layers.Conv2DTranspose(self.base_filters * 4, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same')
        conv4 = ConcreteSpatialDropout2D(conv4_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                         is_mc_dropout=self.is_mc_dropout)(opt_cat_3)
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

        output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', dtype='float32')(conv6)

        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_layer)
