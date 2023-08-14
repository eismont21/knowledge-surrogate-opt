from src.models import CFPNetM
from src.scaler import Scaler
from src.constants import INPUT_SHAPE, OUTPUT_SHAPE, MATRIX_SHAPE

import tensorflow as tf
import numpy as np


class Oracle:
    def __init__(self, model_path="oracle/oracle_128.h5"):
        self.scaler = Scaler()
        with tf.device('/CPU:0'):
            self.model = CFPNetM(name='CFPNetM',
                                 input_dim=INPUT_SHAPE,
                                 output_dim=OUTPUT_SHAPE,
                                 encoding="domain",
                                 positional_encoding=0,
                                 base_filters=128,)
            self.model.reload(is_mc_dropout=False, filepath=model_path)

    def simulate(self, individual, scale_input=True, scale_output=True):
        x = np.array(individual).reshape(1, -1)
        if scale_input:
            x = self.scaler.scale(x, col_name="gripper_force")
        with tf.device('/CPU:0'):
            y = self.model.predict(x)
        if scale_output:
            y = self.scaler.inverse_transform(y, col_name="strain_field_matrix").reshape(MATRIX_SHAPE)
        return y
