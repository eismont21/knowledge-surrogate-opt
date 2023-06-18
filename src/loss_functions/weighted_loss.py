import tensorflow as tf
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE


class WeightedLoss(tf.keras.losses.Loss):
    def __init__(self, obj_function, loss_fn, scale_range=(0.1, 1)):
        super().__init__()
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.scale_range = scale_range

    def call(self, y_true, y_pred):
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function, scale_range=self.scale_range), y_true)

        weighted_loss = self.loss_fn(y_true, y_pred, sample_weight=imp_true)

        return weighted_loss
