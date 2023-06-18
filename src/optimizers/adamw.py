import tensorflow as tf

import tensorflow as tf


class AdamW(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, weight_decay=0.01, epsilon=1e-7, amsgrad=False,
                 name='AdamW', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('weight_decay', weight_decay)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._has_weight_decay = weight_decay != 0.0
        self._initial_weight_decay = weight_decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        m_t = m.assign((self._get_hyper('beta_1', var_dtype) * m) + (1. - self._get_hyper('beta_1', var_dtype)) * grad)
        v_t = v.assign(
            (self._get_hyper('beta_2', var_dtype) * v) + (1. - self._get_hyper('beta_2', var_dtype)) * tf.square(grad))

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = vhat.assign(tf.maximum(vhat, v_t))
            v_sqrt = tf.sqrt(vhat_t)
        else:
            v_sqrt = tf.sqrt(v_t)

        if self._initial_decay > 0:
            lr_t *= (1. - self._get_hyper('beta_2', var_dtype))
        lr_t = lr_t * (tf.sqrt(1. - self._get_hyper('beta_2', var_dtype)) / (1. - self._get_hyper('beta_1', var_dtype)))

        if self._has_weight_decay:
            new_var_m = lr_t * m_t / (v_sqrt + self.epsilon) + lr_t * self._get_hyper('weight_decay', var_dtype) * var
        else:
            new_var_m = lr_t * m_t / (v_sqrt + self.epsilon)
        var_update = var.assign_sub(new_var_m)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        lr_t = lr_t * (tf.sqrt(1. - beta_2_power) / (1. - beta_1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = (1 - beta_1_t) * tf.gather(grad, indices)
        m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (1 - beta_2_t) * tf.gather(tf.square(grad), indices)
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        v_t = v.assign(v * beta_2_t, use_locking=self._use_locking)

        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = v_hat.assign(tf.maximum(v_hat, v_t), use_locking=self._use_locking)
            v_sqrt = tf.sqrt(v_hat_t)
        else:
            v_sqrt = tf.sqrt(v_t)

        var_t = m_t / (v_sqrt + epsilon_t)

        if self._initial_decay > 0:
            var_t += self._initial_decay * var

        var_update = var.assign_sub(lr_t * var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]

        if self.amsgrad:
            updates.append(v_hat_t)

        return tf.group(*updates)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        }
