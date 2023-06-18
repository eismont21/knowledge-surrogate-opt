import tensorflow as tf
import numpy as np
import pandas as pd
import os
from src.scaler import Scaler
from src.constants import STAMP_SHAPE_MATRIX_PATH, ENCODING_PATH, LENGTHS_PATH


class DomainEncoding(tf.keras.layers.Layer):
    def __init__(self, name='domain_encoding', stamp_shape_matrix_path=STAMP_SHAPE_MATRIX_PATH,
                 encodings_path=ENCODING_PATH, lengths_path=LENGTHS_PATH,
                 use_lengths=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.stamp_shape_matrix_path = stamp_shape_matrix_path
        stamp_shape_matrix = np.load(self.stamp_shape_matrix_path)
        stamp_shape_matrix = np.expand_dims(stamp_shape_matrix, axis=-1)
        self.stamp_shape_matrix = tf.convert_to_tensor(stamp_shape_matrix, dtype=tf.float32)
        self.encodings_path = encodings_path
        self.use_lengths = use_lengths
        if self.use_lengths:
            scaler = Scaler()
            self.lenghts = scaler.scale(pd.read_csv(lengths_path).to_numpy(), 'length')

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        encodings = []
        for i in range(self.input_dim):
            encoding = np.load(os.path.join(self.encodings_path, f"encoding_{i}.npy"))
            if self.use_lengths:
                encoding = encoding * self.lenghts[i]
            encodings.append(encoding)

        self.encodings = tf.stack(encodings, axis=-1)
        self.encodings = tf.cast(self.encodings, tf.float32)

        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, [batch_size, 1, 1, -1])

        new_channels = inputs * self.encodings

        stamp_shape_matrix_broadcasted = tf.broadcast_to(
            self.stamp_shape_matrix, [batch_size, *self.stamp_shape_matrix.shape]
        )
        result = tf.concat([stamp_shape_matrix_broadcasted, new_channels], axis=-1)

        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "stamp_shape_matrix_path": self.stamp_shape_matrix_path,
            "encodings_path": self.encodings_path,
        })
        return config
