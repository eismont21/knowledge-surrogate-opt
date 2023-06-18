import tensorflow as tf
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE


class TotalLoss(tf.keras.losses.Loss):
    def __init__(self, obj_function, loss_fn, alpha=0.5):
        super().__init__()
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.alpha = alpha

    def call(self, y_true, y_pred):
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        ml_loss = self.loss_fn(y_true, y_pred)

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_true)
        imp_pred = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_pred)

        importance_ml_loss = self.loss_fn(imp_true, imp_pred)

        return (1 - self.alpha) * ml_loss + self.alpha * importance_ml_loss
