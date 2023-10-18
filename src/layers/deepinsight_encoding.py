import tensorflow as tf
import numpy as np
from src.utils import ImageTransformer
from sklearn.manifold import TSNE
from src.constants import STAMP_SHAPE_MATRIX_PATH
from src.scaler import Scaler


class DeepInsightEncoding(tf.keras.layers.Layer):
    """
    Custom encoding layer that represents data using the DeepInsight+ method. 

    This layer provides various channels, including the deepinsight channel, row-wise copies,
    normalized distance channel, and an equidistant bar graph representation of the input data.

    Attributes:
    - x_train (np.ndarray): Training data used for some transformations.
    - stamp_shape_matrix_path (str): Path to the matrix representing stamp shape.
    - stamp_shape_matrix_shape (Tuple[int, int, int]): Shape of the loaded stamp shape matrix.
    - stamp_shape_matrix (tf.Tensor): Tensor version of the loaded stamp shape matrix.
    
    References:
    - Alok Sharma et al. "DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture" - doi.org/10.1038/s41598-019-47765-6
    - Anuraganand Sharma et al. "Classification with 2-D Convolutional Neural Networks for breast cancer diagnosis" - arXiv:2007.03218
    """

    def __init__(self, x_train: np.ndarray, name: str = 'deepinsight_encoding',
                 stamp_shape_matrix_path: str = STAMP_SHAPE_MATRIX_PATH, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.x_train = x_train
        self.stamp_shape_matrix_path = stamp_shape_matrix_path
        stamp_shape_matrix = np.load(self.stamp_shape_matrix_path)
        stamp_shape_matrix = np.expand_dims(stamp_shape_matrix, axis=-1)
        scaler = Scaler()
        stamp_shape_matrix = scaler.scale(stamp_shape_matrix, col_name="stamp_shape_matrix")
        self.stamp_shape_matrix_shape = stamp_shape_matrix.shape
        self.stamp_shape_matrix = tf.convert_to_tensor(stamp_shape_matrix, dtype=tf.float32)

        distance_metric = 'cosine'
        reducer = TSNE(
            n_components=2,
            metric=distance_metric,
            init='random',
            learning_rate='auto',
            n_jobs=1
        )
        pixel_size = self.stamp_shape_matrix_shape[:-1]
        it = ImageTransformer(
            feature_extractor=reducer,
            pixels=pixel_size)
        it.fit(self.x_train)
        self.coords = it._coords.astype(np.int32)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_dim = input_shape[-1]
        height, width = self.stamp_shape_matrix_shape[:-1]

        base_rows_per_value = tf.math.floordiv(height, self.input_dim)
        extra_rows_values = tf.math.mod(height, self.input_dim)
        self.rows_per_value = tf.ones([self.input_dim], tf.int32) * base_rows_per_value
        self.rows_per_value += tf.sequence_mask(extra_rows_values, self.input_dim, dtype=tf.int32)

        base_cols_per_value = tf.math.floordiv(width, self.input_dim)
        extra_cols_values = tf.math.mod(width, self.input_dim)
        self.cols_per_value = tf.ones([self.input_dim], tf.int32) * base_cols_per_value
        self.cols_per_value += tf.sequence_mask(extra_cols_values, self.input_dim, dtype=tf.int32)

        bar_width = width // (3 * self.input_dim + 2)
        total_gap_width = width - (self.input_dim * bar_width)
        gap_width = total_gap_width // (self.input_dim + 1)
        extra_width = total_gap_width - (gap_width * (self.input_dim + 1))
        beginning_extra_width = extra_width // 2
        end_extra_width = extra_width - beginning_extra_width
        self.bar_width = bar_width
        self.gap_width = gap_width
        self.beginning_extra_width = beginning_extra_width
        self.end_extra_width = end_extra_width

        super().build(input_shape)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        deepinsight_channel = self.deepinsignt(inputs, batch_size)
        row_wise_copy = self.row_wise_copy(inputs, batch_size)
        normalized_distance = self.normalized_distance(inputs)
        equidistant_bar_graph = self.equidistant_bar_graph(inputs, batch_size)

        stamp_shape_matrix_broadcasted = tf.broadcast_to(
            self.stamp_shape_matrix, [batch_size, *self.stamp_shape_matrix.shape]
        )
        result = tf.concat([stamp_shape_matrix_broadcasted, deepinsight_channel, row_wise_copy, normalized_distance,
                            equidistant_bar_graph], axis=-1)

        return result

    def deepinsignt(self, inputs: tf.Tensor, batch_size: int) -> tf.Tensor:
        batch_coords = tf.tile(self.coords, [batch_size, 1])
        batch_coords = tf.reshape(batch_coords, [batch_size, self.coords.shape[0], self.coords.shape[-1]])

        flat_inputs = tf.reshape(inputs, [-1])

        batch_indices = tf.range(batch_size)
        batch_indices = tf.repeat(batch_indices, self.coords.shape[0])
        batch_indices = tf.reshape(batch_indices, [-1, 1])

        indices = tf.concat([batch_indices, tf.reshape(batch_coords, [-1, self.coords.shape[-1]])], axis=1)

        new_channel = tf.scatter_nd(indices, flat_inputs, [batch_size, *self.stamp_shape_matrix_shape[:-1]])
        new_channel = tf.expand_dims(new_channel, axis=-1)
        return new_channel

    def row_wise_copy(self, inputs: tf.Tensor, batch_size: int) -> tf.Tensor:
        repeated_indices = tf.repeat(tf.range(self.input_dim), self.rows_per_value)

        inputs_expanded = tf.gather(inputs, repeated_indices, axis=1)

        new_channel = tf.reshape(inputs_expanded, [batch_size, -1, 1])
        new_channel = tf.broadcast_to(new_channel, [batch_size, *self.stamp_shape_matrix_shape[:-1]])
        new_channel = tf.expand_dims(new_channel, axis=-1)
        return new_channel

    def normalized_distance(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs_expanded = tf.expand_dims(inputs, axis=2)
        batched_distance_matrices = tf.math.abs(inputs_expanded - tf.transpose(inputs_expanded, perm=[0, 2, 1]))

        min_val = tf.reduce_min(batched_distance_matrices, axis=[1, 2], keepdims=True)
        max_val = tf.reduce_max(batched_distance_matrices, axis=[1, 2], keepdims=True)
        normalized_matrices = (batched_distance_matrices - min_val) / (max_val - min_val)

        expanded_rows = tf.repeat(normalized_matrices, self.rows_per_value, axis=1)
        expanded_matrix = tf.repeat(expanded_rows, self.cols_per_value, axis=2)

        new_channel = tf.expand_dims(expanded_matrix, axis=-1)
        return new_channel

    def equidistant_bar_graph(self, inputs: tf.Tensor, batch_size: int) -> tf.Tensor:
        height, width = self.stamp_shape_matrix_shape[:-1]

        bar_plot = tf.zeros((batch_size, height, width), dtype=inputs.dtype)

        for i in range(self.input_dim):
            start_index = self.beginning_extra_width + i * (self.bar_width + self.gap_width) + self.gap_width
            end_index = start_index + self.bar_width

            bar_height = tf.cast(tf.round(inputs[:, i:i + 1] * height), dtype=tf.int32)
            bar_height = tf.clip_by_value(bar_height, 0, height)

            mask = tf.sequence_mask(bar_height, maxlen=height)
            mask = tf.cast(mask, dtype=inputs.dtype)
            mask = tf.transpose(mask, [0, 2, 1])

            bar_mask = tf.zeros((batch_size, height, width), dtype=inputs.dtype)
            for j in range(start_index, end_index):
                bar_mask += tf.where(
                    tf.expand_dims(tf.range(width), 0) == j,
                    tf.ones((batch_size, height, 1), dtype=inputs.dtype),
                    0
                )

            bar_plot = bar_plot * (1 - bar_mask) + mask * bar_mask

        new_channel = tf.expand_dims(bar_plot, axis=-1)

        return new_channel

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "stamp_shape_matrix_path": self.stamp_shape_matrix_path,
        })
        return config
