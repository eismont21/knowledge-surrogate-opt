from src.models import Model
import tensorflow as tf
from src.layers.concrete_dropout import ConcreteDenseDropout, get_weight_regularizer, get_dropout_regularizer


class DenseModel(Model):
    """
    DenseModel is an implementation of the FCNN model.

    References:
    - Clemens Zimmerling et al. "Deep neural networks as surrogate models for time-efficient manufacturing process optimisation" - DOI:10.25518/esaform21.3882

    Attributes:
    - hidden_neurons (int): Number of neurons in the hidden layers.
    - activation (str): Activation function used in the network.
    """

    def __init__(self, name: str, input_dim, output_dim, hidden_neurons: int = 500,
                 activation: str = 'relu') -> None:
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        super().__init__(name, input_dim, output_dim)

    def build(self) -> None:
        input_tensor = tf.keras.Input(shape=(self.input_dim[0],))
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(input_tensor)
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(x)
        x = tf.keras.layers.Dense(self.hidden_neurons, activation=self.activation)(x)
        output_layer = tf.keras.layers.Dense(self.output_dim[0], activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)


class DenseModelDropout(Model):
    """
    DenseModelDropout is an implementation of the FCNN model with concrete dropout functionality.

    References:
    - Clemens Zimmerling et al. "Deep neural networks as surrogate models for time-efficient manufacturing process optimisation" - DOI:10.25518/esaform21.3882

    Attributes:
    - hidden_neurons (int): Number of neurons in the hidden layers.
    - activation (str): Activation function for the hidden layers.
    - wr: Weight regularization parameter.
    - dr: Dropout regularization parameter.
    - is_mc_dropout (bool): If True, applies Monte Carlo dropout.
    """

    def __init__(self, name: str, input_dim, output_dim, train_size: int = 100, hidden_neurons: int = 500,
                 activation: str = 'relu', is_mc_dropout: bool = False) -> None:
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.wr = get_weight_regularizer(train_size, l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(train_size, tau=1.0)
        self.is_mc_dropout = is_mc_dropout
        super().__init__(name, input_dim, output_dim)

    def set_train_size(self, train_size: int) -> None:
        self.wr = get_weight_regularizer(train_size, l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(train_size, tau=1.0)
        for layer in self.model.layers:
            if isinstance(layer, ConcreteDenseDropout):
                layer.set_regularizers(self.wr, self.dr)

    def build(self) -> None:
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
