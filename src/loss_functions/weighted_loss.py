import tensorflow as tf
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE
from typing import Tuple, Callable


class WeightedLoss(tf.keras.losses.Loss):
    """
    Loss function that calculates a weighted loss using an importance matrix.

    The WeightedLoss class computes a loss value between true and predicted values based on an objective
    function and a standard loss function, all while considering an importance matrix that weighs the contributions
    of different elements in the matrices.

    Attributes:
    - obj_function (Callable): The objective function used to compute the importance matrix.
    - loss_fn (Callable): The standard loss function to which the importance matrix will be applied.
    - scale_range (Tuple[float, float]): A tuple indicating the range to scale the importance matrix.
    """

    def __init__(self, obj_function: Callable, loss_fn: Callable, scale_range: Tuple[float, float] = (0.1, 1)) -> None:
        super().__init__()
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.scale_range = scale_range

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function, scale_range=self.scale_range), y_true)

        weighted_loss = self.loss_fn(y_true, y_pred, sample_weight=imp_true)

        return weighted_loss
