import tensorflow as tf
import numpy as np
from src.constants import STAMP_SHAPE_MATRIX_PATH
from src.scaler import Scaler


class NaiveEncoding(tf.keras.layers.Layer):
    def __init__(self, name='naive_encoding', stamp_shape_matrix_path=STAMP_SHAPE_MATRIX_PATH,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.stamp_shape_matrix_path = stamp_shape_matrix_path
        stamp_shape_matrix = np.load(self.stamp_shape_matrix_path)
        stamp_shape_matrix = np.expand_dims(stamp_shape_matrix, axis=-1)
        scaler = Scaler()
        stamp_shape_matrix = scaler.scale(stamp_shape_matrix, col_name="stamp_shape_matrix")
        self.stamp_shape_matrix = tf.convert_to_tensor(stamp_shape_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        super().build(input_shape)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, [batch_size, 1, 1, -1])

        input_shape = [batch_size, tf.shape(self.stamp_shape_matrix)[0], tf.shape(self.stamp_shape_matrix)[1],
                       self.input_dim]
        new_channels = tf.broadcast_to(inputs, input_shape)

        stamp_shape_matrix_broadcasted = tf.broadcast_to(
            self.stamp_shape_matrix, [batch_size, *self.stamp_shape_matrix.shape]
        )
        result = tf.concat([stamp_shape_matrix_broadcasted, new_channels], axis=-1)

        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "stamp_shape_matrix_path": self.stamp_shape_matrix_path,
        })
        return config