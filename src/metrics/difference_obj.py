import tensorflow as tf
from src.scaler import Scaler
from typing import Callable


class DifferenceObjectiveFunction(tf.keras.metrics.Metric):
    """
    Custom Keras metric to compute the difference of objective functions between predicted and true values.

    This metric allows for optional inverse transformation of the inputs before computing
    the objective function differences. It maintains a running sum of the differences and
    the count of processed batches to compute the average difference over batches.

    Attributes:
    - sum (tf.Variable): Ssum of the difference between predicted and true objective function values.
    - count (tf.Variable): Total number of elements used for calculating.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    - obj_function (Callable): Objective function to evaluate the inputs.
    """

    def __init__(self, obj_function: Callable, inverse: bool = False, name: str = 'difference_obj', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.scaler = Scaler()
        self.inverse = inverse
        self.obj_function = obj_function

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        y_true_obj = tf.map_fn(self.obj_function, y_true)
        y_pred_obj = tf.map_fn(self.obj_function, y_pred)

        error = y_pred_obj - y_true_obj
        squared_sum = tf.reduce_sum(error)
        count = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.sum.assign_add(squared_sum)
        self.count.assign_add(count)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.sum, self.count)

    def reset_state(self) -> None:
        self.sum.assign(0.0)
        self.count.assign(0.0)
