import tensorflow as tf
from src.scaler import Scaler
from src.constants import MATRIX_SHAPE


class SSIMLossMetric(tf.keras.metrics.Metric):
    """
    Custom metric to compute the Structural Similarity Index (SSIM) loss.

    Attributes:
    - max_val (int): The maximum value of the data type for SSIM computation.
    - filter_size (int): The size of the window for SSIM computation.
    - total (tf.Tensor): Sum of SSIM loss values.
    - count (tf.Tensor): Total number of elements used for calculating SSIM loss.
    - scaler (Scaler): Instance of the Scaler class to inverse transform values.
    - inverse (bool): If True, inverse transformation is applied to the inputs.
    """

    def __init__(self, name: str = 'loss_ssim', inverse: bool = False, max_val: int = 90, filter_size: int = 5,
                 **kwargs) -> None:
        super(SSIMLossMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.filter_size = filter_size
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.scaler = Scaler()
        self.inverse = inverse
        if not self.inverse:
            self.max_val = self.scaler.scale(self.max_val, col_name="strain_field_matrix")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        if self.inverse:
            y_true = self.scaler.inverse_transform(y_true, col_name="strain_field_matrix")
            y_pred = self.scaler.inverse_transform(y_pred, col_name="strain_field_matrix")

        ssim_value = 1 - tf.image.ssim(y_true, y_pred, max_val=self.max_val, filter_size=self.filter_size)
        self.total.assign_add(tf.reduce_sum(ssim_value))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self) -> None:
        self.total.assign(0.)
        self.count.assign(0.)
