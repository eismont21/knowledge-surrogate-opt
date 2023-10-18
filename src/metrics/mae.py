import tensorflow as tf
from src.scaler import Scaler


class MAE(tf.keras.metrics.Metric):
    """
    Custom Mean Absolute Error (MAE) metric with optional inverse scaling.

    Attributes:
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    - abs_sum (tf.Tensor): Sum of absolute differences between predictions and true values.
    - count (tf.Tensor): Total number of elements used for calculating MAE.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    """

    def __init__(self, inverse: bool = False, name: str = 'mae', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.abs_sum = self.add_weight(name="abs_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.scaler = Scaler()
        self.inverse = inverse

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        error = tf.abs(y_pred - y_true)
        abs_sum = tf.reduce_sum(error)
        count = tf.cast(tf.size(y_true), dtype=tf.float32)

        self.abs_sum.assign_add(abs_sum)
        self.count.assign_add(count)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.abs_sum, self.count)

    def reset_state(self) -> None:
        self.abs_sum.assign(0.0)
        self.count.assign(0.0)
