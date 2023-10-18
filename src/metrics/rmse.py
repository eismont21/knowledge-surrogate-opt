import tensorflow as tf
from src.scaler import Scaler
from typing import Callable


class RMSE(tf.keras.metrics.Metric):
    """
    Custom Root Mean Squared Error (RMSE) metric with optional inverse scaling.

    Attributes:
    - squared_sum (tf.Tensor): Sum of squared errors between true and predicted values.
    - count (tf.Tensor): Total number of elements used for calculating RMSE.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    - squared (bool): If True, returns MSE.
    - obj_function (Callable | None): An optional objective function to evaluate the inputs.
    """

    def __init__(self, obj_function: Callable | None = None, inverse: bool = False, name: str = 'rmse',
                 squared: bool = False, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight(name="squared_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.scaler = Scaler()
        self.inverse = inverse
        self.squared = squared
        self.obj_function = obj_function

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        if self.obj_function is not None:
            y_true = tf.map_fn(self.obj_function, y_true)
            y_pred = tf.map_fn(self.obj_function, y_pred)
            count = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        else:
            count = tf.cast(tf.size(y_true), dtype=tf.float32)

        error = tf.math.squared_difference(y_pred, y_true)
        squared_sum = tf.reduce_sum(error)

        self.squared_sum.assign_add(squared_sum)
        self.count.assign_add(count)

    def result(self) -> tf.Tensor:
        mse = tf.math.divide_no_nan(self.squared_sum, self.count)
        if self.squared:
            return mse
        else:
            return tf.sqrt(mse)

    def reset_state(self) -> None:
        self.squared_sum.assign(0.0)
        self.count.assign(0.0)
