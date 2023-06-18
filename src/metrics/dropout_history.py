import keras.backend as K
import tensorflow as tf
import numpy as np


class DropoutHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(DropoutHistory, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ps = np.array([K.eval(layer.p_logit) for layer in self.model.layers if hasattr(layer, 'p_logit')])
        dropout_val = tf.nn.sigmoid(ps).numpy().reshape(-1)
        for i, val in enumerate(dropout_val):
            logs[f'dropout_rate_layer_{i + 1}'] = val
