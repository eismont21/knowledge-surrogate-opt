import tensorflow as tf
from typing import Callable, Tuple


def importance_matrix(matrix: tf.Tensor, obj_function: Callable, scale_range: Tuple[float, float] = (0, 1),
                      eps: float = 1e-7) -> tf.Tensor:
    """
    Computes the importance matrix of a given matrix using its gradients
    with respect to a given objective function.

    The importance is computed as normalized gradients, and then rescaled
    within a given scale range.

    Args:
    - matrix (tf.Tensor): The input matrix.
    - obj_function (Callable): The objective function.
    - scale_range (Tuple[float, float], optional): The range to which the
                                                   normalized gradients should
                                                   be rescaled. Default is
                                                   (0, 1).
    - eps (float): A small constant to prevent division by zero or
                   to replace NaN or zero gradients. Default is 1e-7.

    Returns:
    - tf.Tensor: The rescaled importance matrix with the same shape as the
                 input matrix.
    """
    flat_matrix = tf.reshape(matrix, [-1])

    with tf.GradientTape() as tape:
        tape.watch(flat_matrix)
        value = obj_function(flat_matrix)

    grads = tape.gradient(value, flat_matrix)
    grads = tf.where(tf.math.is_nan(grads), eps, grads)
    grads = tf.where(tf.math.equal(grads, 0), eps, grads)

    min_grads = tf.reduce_min(grads)
    max_grads = tf.reduce_max(grads)
    normalized_grads = (grads - min_grads) / (max_grads - min_grads + eps)

    rescaled_grads = (normalized_grads * (scale_range[1] - scale_range[0])) + scale_range[0]

    return tf.reshape(rescaled_grads, tf.shape(matrix))
