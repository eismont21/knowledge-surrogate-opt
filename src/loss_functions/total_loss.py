import tensorflow as tf
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE
from typing import Callable


class TotalLoss(tf.keras.losses.Loss):
    """
    Domain informed loss function that combines classical ML loss with importance-adjusted loss.

    The TotalLoss class calculates the combined loss by weighting the classical loss (ml_loss)
    with an importance-adjusted version (importance_ml_loss) based on a provided objective function.

    Attributes:
    - obj_function (Callable): The objective function used to compute the importance matrix.
    - loss_fn (Callable): A TensorFlow loss function (e.g., mean squared error).
    - alpha (float): Weight for the importance-adjusted loss in the final combined loss.
    """

    def __init__(self, obj_function: Callable, loss_fn: Callable, alpha: float = 0.5) -> None:
        super().__init__()
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        ml_loss = self.loss_fn(y_true, y_pred)

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_true)
        imp_pred = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_pred)

        importance_ml_loss = self.loss_fn(imp_true, imp_pred)

        return (1 - self.alpha) * ml_loss + self.alpha * importance_ml_loss
