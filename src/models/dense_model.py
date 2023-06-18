from src.models import Model
import tensorflow as tf
from src.layers.concrete_dropout import ConcreteDenseDropout, get_weight_regularizer, get_dropout_regularizer


class DenseModel(Model):
    def __init__(self, name: str, input_dim, output_dim, hidden_neurons: int = 500,
                 activation: str = 'relu'):
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        super().__init__(name, input_dim, output_dim)

    def build(self):
        input_tensor = tf.keras.Input(shape=(self.input_dim[0],))
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(input_tensor)
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(x)
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(x)
        output_layer = tf.keras.layers.Dense(self.output_dim[0], activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)


class DenseModelDropout(Model):
    def __init__(self, name: str, input_dim, output_dim, x_train, hidden_neurons: int = 500,
                 activation: str = 'relu', is_mc_dropout: bool = False):
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.wr = get_weight_regularizer(x_train.shape[0], l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(x_train.shape[0], tau=1.0)
        self.is_mc_dropout = is_mc_dropout
        super().__init__(name, input_dim, output_dim)

    def build(self):
        input_tensor = tf.keras.Input(shape=(self.input_dim[0],))

        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(input_tensor)

        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        x = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(x)

        dense_layer = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)
        x = ConcreteDenseDropout(dense_layer, weight_regularizer=self.wr, dropout_regularizer=self.dr,
                                 is_mc_dropout=self.is_mc_dropout)(x)

        output_layer = tf.keras.layers.Dense(self.output_dim[0], activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
