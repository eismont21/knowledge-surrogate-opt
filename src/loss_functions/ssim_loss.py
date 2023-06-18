import tensorflow as tf
from src.constants import MATRIX_SHAPE


class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, max_val=1.0, filter_size=5):
        super().__init__()
        self.max_val = max_val
        self.filter_size = filter_size

    def call(self, y_true, y_pred):
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val, filter_size=self.filter_size))
