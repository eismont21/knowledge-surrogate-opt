import tensorflow as tf
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    """
    A layer that computes a classical positional encoding for an input tensor.

    The positional encoding is computed based on the idea that the position of a token
    in a sequence can impact its semantics. The encoding utilizes sinusoidal functions
    to produce a unique encoding for each position.

    The encoding produced is added to the input embeddings to give the model
    information about the relative positions of tokens.

    Source:
    - https://github.com/tatp22/multidim-positional-encoding

    References:
    - Ashish Vaswani et al. "Attention Is All You Need." - arXiv:1706.03762
    - Zelun Wang et al. “Translating math formula images to LaTeX sequences using deep neural networks with sequence-level training” - doi.org/10.1007/s10032-020-00360-2

    Attributes:
    - inv_freq (np.ndarray): Inverse frequencies used to calculate the sinusoidal encodings.
    """

    def __init__(self, name: str = 'positional_encoding', **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = int(2 * np.ceil(input_shape[-1] / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, channels, 2) / np.float32(channels)
            )
        )
        super().build(input_shape)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
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
