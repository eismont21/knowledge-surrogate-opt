import tensorflow as tf
from src.utils import importance_matrix
from src.constants import MATRIX_SHAPE


class TotalLossMetric(tf.keras.metrics.Metric):
    def __init__(self, obj_function, loss_fn, name="total_loss", alpha=0.5, **kwargs):
        super(TotalLossMetric, self).__init__(name=name, **kwargs)
        self.obj_function = obj_function
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(tf.shape(y_true)) != 4:
            y_true = tf.reshape(y_true, [-1, *MATRIX_SHAPE, 1])
            y_pred = tf.reshape(y_pred, [-1, *MATRIX_SHAPE, 1])

        ml_loss = self.loss_fn(y_true, y_pred)

        imp_true = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_true)
        imp_pred = tf.map_fn(lambda y: importance_matrix(y, self.obj_function), y_pred)

        importance_ml_loss = self.loss_fn(imp_true, imp_pred)

        total_loss = (1 - self.alpha) * ml_loss + self.alpha * importance_ml_loss
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.total.assign_add(batch_size * total_loss)
        self.count.assign_add(batch_size)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)
