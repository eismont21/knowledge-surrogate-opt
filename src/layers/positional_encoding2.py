import tensorflow as tf
import numpy as np


class PositionalEncoding2(tf.keras.layers.Layer):
    """
    A layer that computes a optimized positional encoding for an input tensor.

    The positional encoding is computed based on the idea that the position of a token
    in a sequence can impact its semantics. The encoding utilizes sinusoidal functions
    to produce a unique encoding for each position. Drawing on insights regarding the
    draping process, particularly about the known positions of the grippers around the
    textile, we devise an encoding where the peaks of the sin and cos functions
    coincide, to an approximation, with the grippers' positions.

    The encoding produced is concatenated to the input embeddings to give the model
    information about the relative positions of tokens.

    Source:
    - https://github.com/tatp22/multidim-positional-encoding

    References:
    - Ashish Vaswani et al. "Attention Is All You Need." - arXiv:1706.03762
    - Zelun Wang et al. “Translating math formula images to LaTeX sequences using deep neural networks with sequence-level training” - doi.org/10.1007/s10032-020-00360-2

    Attributes:
    - channels (int): Number of channels for the positional encoding to achieve the desired peaks locations.
    - inv_freq (np.ndarray): Inverse frequencies used to calculate the sinusoidal encodings.
    """

    def __init__(self, channels: int = 36, name: str = 'positional_encoding_v2', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        _, x, y, _ = inputs.shape

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
            emb[None, :, :, :], tf.shape(inputs)[0], axis=0
        )
        return tf.concat([inputs, cached_penc[..., 3:12], cached_penc[..., 21:30]], axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        return config

    @staticmethod
    def get_emb(sin_inp: tf.Tensor) -> tf.Tensor:
        """
        Computes the base embedding for a given dimension using sinusoidal functions.

        The embedding alternates between sine and cosine values.

        Args:
        - sin_inp (tf.Tensor): Sinusoidal input tensor.

        Returns:
        - tf.Tensor: Computed embedding.
        """
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        return emb
