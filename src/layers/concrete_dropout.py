from typing import Union, Tuple
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Wrapper, InputSpec


def get_weight_regularizer(N: int, l: float = 1e-2, tau: float = 0.1) -> float:
    """
    Compute the weight regularizer for Concrete Dropout.

    The weight regularizer is calculated as the square of the lengthscale divided
    by the product of model precision (tau) and the number of data points (N).

    Args:
    - N (int): Number of instances in the dataset.
    - l (float, optional): Prior lengthscale. Default is 1e-2.
    - tau (float, optional): Model precision (inverse observation noise). Default is 0.1.

    Returns:
    float: Computed weight regularizer.
    """
    return l ** 2 / (tau * N)


def get_dropout_regularizer(N: int, tau: float = 0.1, cross_entropy_loss: bool = False) -> float:
    """
    Compute the dropout regularizer for Concrete Dropout.

    The dropout regularizer is calculated based on the number of data points (N),
    model precision (tau), and whether the loss function is cross-entropy or not.

    Args:
    - N (int): Number of instances in the dataset.
    - tau (float, optional): Model precision (inverse observation noise). Default is 0.1.
    - cross_entropy_loss (bool, optional): If True, factor of two is ignored. Default is False.

    Returns:
    float: Computed dropout regularizer.
    """
    reg = 1 / (tau * N)
    if not cross_entropy_loss:
        reg *= 2
    return reg


class ConcreteDropout(Wrapper):
    """
    Base class for ConcreteDropout. Allows learning the dropout probability for a given layer.
    These layers perform dropout BEFORE the wrapped operation.

    ConcreteDropout is a form of dropout where the dropout probability is learned
    as a parameter during training. It is based on the Concrete (or Gumbel-Softmax)
    distribution.

    Attributes:
    - layer (Layer): A tensorflow.keras layer instance.
    - weight_regularizer (float, optional): Regularizer for layer weights. Default is 1e-6.
    - dropout_regularizer (float, optional): Regularizer for dropout. Default is 1e-5.
    - init_min (float, optional): Initial minimum value for dropout probability in logit space. Default is 0.1.
    - init_max (float, optional): Initial maximum value for dropout probability in logit space. Default is 0.1.
    - is_mc_dropout (bool, optional): If True, apply Monte Carlo dropout, else apply standard dropout. Default is False.
    - data_format (str, optional): Data format, either 'channels_last' or 'channels_first'. Default is 'channels_last'.
    - temperature (float, optional): Temperature of Concrete Distribution. Determines the "sharpness" of the learned dropout. Default is 0.1.

    References:
    Yarin Gal et al. "Concrete Dropout" - arXiv:1705.07832
    """

    def __init__(self, layer: tf.keras.layers.Layer, weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5, init_min: float = 0.1,
                 init_max: float = 0.1, is_mc_dropout: bool = False,
                 data_format: Union[str, None] = None, temperature: float = 0.1, **kwargs) -> None:
        assert 'kernel_regularizer' not in kwargs, "Must not provide a kernel regularizer."
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = tf.math.log(init_min) - tf.math.log(1. - init_min)
        self.init_max = tf.math.log(init_max) - tf.math.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(
                                           self.init_min, self.init_max),
                                       dtype=tf.dtypes.float32,
                                       trainable=True)

    def set_regularizers(self, weight_regularizer: float, dropout_regularizer: float) -> None:
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        initial_value = tf.random_uniform_initializer(self.init_min, self.init_max)(shape=(1,))
        self.p_logit.assign(initial_value)

    def _get_noise_shape(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError(
            "Subclasses of ConcreteDropout must implement the noise shape")

    def compute_output_shape(self, input_shape: tf.Tensor) -> tf.Tensor:
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        eps = K.cast_to_floatx(K.epsilon())

        noise_shape = self._get_noise_shape(x)

        unif_noise = K.random_uniform(shape=noise_shape)
        drop_prob = (
                tf.math.log(p + eps)
                - tf.math.log1p(eps - p)
                + tf.math.log(unif_noise + eps)
                - tf.math.log1p(eps - unif_noise)
        )
        drop_prob = tf.math.sigmoid(drop_prob / self.temperature)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor
        x /= retain_prob

        return x

    def call(self, inputs: tf.Tensor, training: Union[bool, None] = None) -> tf.Tensor:
        p = tf.nn.sigmoid(self.p_logit)
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p)
        dropout_regularizer += (1. - p) * tf.math.log1p(- p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        self.layer.add_loss(regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs, p))

            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


class ConcreteDenseDropout(ConcreteDropout):
    """
    Wrapper for Dense layers to learn the dropout probability.

    This class is a specialized version of the ConcreteDropout layer, tailored for dense layers.

    Attributes:
    - layer (Layer): A tensorflow.keras dense layer instance.
    - temperature (float, optional): Temperature of Concrete Distribution. Default is 0.1.

    Example:
    ```python
    x = # some input layer
    x = ConcreteDenseDropout(Dense(64))(x)
    ```
    """

    def __init__(self, layer: tf.keras.layers.Layer, temperature: float = 0.1, **kwargs) -> None:
        super(ConcreteDenseDropout, self).__init__(
            layer, temperature=temperature, **kwargs)

    def _get_noise_shape(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        return input_shape

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteDenseDropout, self).build(input_shape=input_shape)

        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        self.input_dim = input_shape[1]

    def call(self, inputs: tf.Tensor, training: Union[bool, None] = None) -> tf.Tensor:
        return super().call(inputs, training=training)


class ConcreteSpatialDropout2D(ConcreteDropout):
    """
    Wrapper for 2D convolutional layers to learn the spatial dropout probability.

    This class is a specialized version of the ConcreteDropout layer, tailored for Conv2D layers, and
    implements spatial dropout, where entire channels (rather than individual units) are dropped out.

    Attributes:
    - layer (Layer): A tensorflow.keras Conv2D layer instance.
    - temperature (float, optional): Temperature of Concrete Distribution. Default is 2/3.

    Example:
    ```python
    x = # some input layer
    x = ConcreteSpatialDropout2D(Conv2D(64, (3,3)))(x)
    ```
    """

    def __init__(self, layer: tf.keras.layers.Layer, temperature: float = 2 / 3, **kwargs) -> None:
        super(ConcreteSpatialDropout2D, self).__init__(
            layer, temperature=temperature, **kwargs)

    def _get_noise_shape(self, inputs: tf.Tensor) -> Tuple[int, int, int, int]:
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[3])

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_spec = InputSpec(shape=input_shape)

        super(ConcreteSpatialDropout2D, self).build(input_shape=input_shape)

        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
        else:
            self.input_dim = input_shape[3]

    def call(self, inputs: tf.Tensor, training: Union[bool, None] = None) -> tf.Tensor:
        return super().call(inputs, training=training)
