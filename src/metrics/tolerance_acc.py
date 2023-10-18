import tensorflow as tf
from src.scaler import Scaler


class ToleranceAccuracy(tf.keras.metrics.Metric):
    """
    Custom metric to measure the accuracy of predictions within a certain tolerance.

    Attributes:
    - tolerance (float): The acceptable difference between true and predicted values.
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    - acc (tf.Tensor): Sum of accuracies.
    - count (tf.Tensor): otal number of elements used for calculating the accuracy.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    """

    def __init__(self, tolerance: float, inverse: bool = False, name: str = 'tolerance_accuracy', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.acc = self.add_weight(name="acc", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.scaler = Scaler()
        self.tolerance = tolerance
        self.inverse = inverse
        if not self.inverse:
            self.tolerance = self.scaler.scale(self.tolerance, col_name="strain_field_matrix")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        error = tf.abs(y_pred - y_true)
        correct = tf.cast(tf.less_equal(error, self.tolerance), tf.float32)
        correct_count = tf.reduce_sum(correct, axis=1)
        vector_size = tf.cast(tf.shape(y_true)[1], dtype=tf.float32)
        accuracy = tf.math.divide_no_nan(correct_count, vector_size)
        sum_acc = tf.reduce_sum(accuracy)
        count = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.acc.assign_add(sum_acc)
        self.count.assign_add(count)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.acc, self.count)

    def reset_state(self) -> None:
        self.acc.assign(0.0)
        self.count.assign(0.0)
