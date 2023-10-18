import tensorflow as tf
import numpy as np
import pandas as pd
import os
from src.scaler import Scaler
from src.constants import STAMP_SHAPE_MATRIX_PATH, ENCODING_PATH, LENGTHS_PATH


class DomainEncoding(tf.keras.layers.Layer):
    """
    Custom layer for domain-based encodings (Stiffness Changes and Force Changes).

    This method lies on the transformation equation that explicate the changes in stiffness ensuing from
    the rotation of fibers relative to the force application point.

    Attributes:
    - stamp_shape_matrix_path (str): Path to the stamp shape matrix data.
    - encodings_path (str): Path to the precomputed domain encodings.
    - lengths_path (str): Path to data of the lengths of stretched springs.
    - use_lengths (bool): Flag to determine if lengths are used for scaling encodings (Force Changes Encoding).
    - stamp_shape_matrix (tf.Tensor): Tensor representation of the stamp shape matrix.
    - encodings (tf.Tensor): Tensor representation of the domain encodings.
    - lengths (np.ndarray): If use_lengths is True, contains the scaled lengths.
    """

    def __init__(self, name: str = 'domain_encoding', stamp_shape_matrix_path: str = STAMP_SHAPE_MATRIX_PATH,
                 encodings_path: str = ENCODING_PATH, lengths_path: str = LENGTHS_PATH, use_lengths: bool = False,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.stamp_shape_matrix_path = stamp_shape_matrix_path
        stamp_shape_matrix = np.load(self.stamp_shape_matrix_path)
        stamp_shape_matrix = np.expand_dims(stamp_shape_matrix, axis=-1)
        scaler = Scaler()
        stamp_shape_matrix = scaler.scale(stamp_shape_matrix, col_name="stamp_shape_matrix")
        self.stamp_shape_matrix = tf.convert_to_tensor(stamp_shape_matrix, dtype=tf.float32)
        self.encodings_path = encodings_path
        self.use_lengths = use_lengths
        if self.use_lengths:
            self.lenghts = scaler.scale(pd.read_csv(lengths_path).to_numpy(), col_name="length")

    def build(self, input_shape: tf.TensorShape) -> None:
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
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, [batch_size, 1, 1, -1])

        new_channels = inputs * self.encodings

        stamp_shape_matrix_broadcasted = tf.broadcast_to(
            self.stamp_shape_matrix, [batch_size, *self.stamp_shape_matrix.shape]
        )
        result = tf.concat([stamp_shape_matrix_broadcasted, new_channels], axis=-1)

        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "stamp_shape_matrix_path": self.stamp_shape_matrix_path,
            "encodings_path": self.encodings_path,
        })
        return config
