import tensorflow as tf


def importance_matrix(matrix, obj_function, scale_range=(0, 1), eps=1e-7):
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
