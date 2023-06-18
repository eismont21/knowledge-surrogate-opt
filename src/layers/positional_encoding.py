import tensorflow as tf
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Source: https://github.com/tatp22/multidim-positional-encoding
    """
    def __init__(self, name='positional_encoding', **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        channels = int(2 * np.ceil(input_shape[-1] / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, channels, 2) / np.float32(channels)
            )
        )
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(self.get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(self.get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        cached_penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        outputs = inputs + cached_penc
        return outputs

    def get_config(self):
        config = super().get_config()
        return config

    @staticmethod
    def get_emb(sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        return emb