import tensorflow as tf
from src.constants import MATRIX_SHAPE


class SSIMLoss(tf.keras.losses.Loss):
    """
    Implements the Structural Similarity Index (SSIM) as a loss function.

    SSIM is a metric used to measure the structural similarity between two images.
    This class provides a loss based on the dissimilarity (1 - SSIM) between predictions
    and true values, suitable for training neural networks.

    Attributes:
    - max_val (float): The dynamic range of the images (i.e., the difference between the maximum
                         the allowed value and the minimum allowed value).
    - filter_size (int): Size of the window for SSIM calculation.
    """

    def __init__(self, max_val: float = 1.0, filter_size: int = 5) -> None:
        super().__init__()
        self.max_val = max_val
        self.filter_size = filter_size

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val, filter_size=self.filter_size))
