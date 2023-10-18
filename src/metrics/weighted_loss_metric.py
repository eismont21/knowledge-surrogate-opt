import tensorflow as tf
from src.scaler import Scaler
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE
from typing import Callable, Tuple


class WeightedLossMetric(tf.keras.metrics.Metric):
    """
    Custom metric that computes a weighted loss using importance matrices.

    Attributes:
    - obj_function (Callable): Objective function used to compute the importance matrix.
    - loss_fn (Callable): Original loss function to which the importance weighting is applied.
    - scale_range (Tuple[float, float]): Range for scaling the importance matrix values.
    - total (tf.Tensor): Accumulator for the total weighted loss across batches.
    - count (tf.Tensor): Total number of elements used for calculating the weighted loss.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    """

    def __init__(self, obj_function: Callable, loss_fn: Callable, inverse: bool = False, name: str = "weighted_loss",
                 scale_range: Tuple[float, float] = (0.1, 1), **kwargs) -> None:
        super(WeightedLossMetric, self).__init__(name=name, **kwargs)
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.scale_range = scale_range
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.scaler = Scaler()
        self.inverse = inverse

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function, scale_range=self.scale_range), y_true)

        weighted_loss = self.loss_fn(y_true, y_pred, sample_weight=imp_true)
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.total.assign_add(batch_size * weighted_loss)
        self.count.assign_add(batch_size)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.)
        self.count.assign(0.)
