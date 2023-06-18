from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from src.scaler import Scaler


class DataLoader(ABC):
    def __init__(self, data_path: str, input_regex: str, output_regex: str):
        self.data_path = os.path.abspath(data_path)
        self.input_regex = input_regex
        self.output_regex = output_regex
        self.scaler = Scaler()

    def load_data(self):
        data = pd.read_csv(self.data_path)
        x = data.filter(regex=self.input_regex).copy(deep=True)
        y = data.filter(regex=self.output_regex).copy(deep=True)
        return x, y

    @abstractmethod
    def create_dataset(self, x, y, batch_size):
        pass

    @staticmethod
    def split_data(x, y, train_size=None, test_size=None, train_ratio=None, test_ratio=None, seed=42, shuffle=True):

        if train_size is not None and test_size is not None:
            assert train_size + test_size <= len(
                x), "Train size and Test size should be less than or equal to the total dataset size"
        else:
            assert train_ratio is not None and test_ratio is not None, "Either absolute size or ratio should be provided"
            assert train_ratio + test_ratio <= 1, "Train ratio and Test ratio should add up to 1 or less"
            train_size = int(train_ratio * len(x.index))
            test_size = int(test_ratio * len(x.index))

        np.random.seed(seed)

        if shuffle:
            indices = np.random.permutation(x.index)
        else:
            indices = x.index.to_numpy()

        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]

        x_train, x_test = x.loc[train_indices], x.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]

        return x_train, x_test, y_train, y_test


class BaselineDataLoader(DataLoader):
    def __init__(self, data_path, input_regex="gripper_force", output_regex="strain_field_matrix_path"):
        super().__init__(data_path, input_regex, output_regex)

    def create_dataset(self, x, y, batch_size, shuffle=True):
        x_scaled = self.scaler.scale(x)

        y_copy = y.copy(deep=True)
        y_copy['strain_field_matrix'] = y_copy['strain_field_matrix_path'].apply(
            lambda path: np.load(path).flatten()
        )
        y_scaled = self.scaler.scale(y_copy)
        y_scaled.drop(columns=['strain_field_matrix_path'], inplace=True)
        y_scaled = pd.DataFrame(y_scaled['strain_field_matrix'].values.tolist(), index=y_scaled.index)

        dataset = tf.data.Dataset.from_tensor_slices((x_scaled.values.astype(np.float32),
                                                      y_scaled.values.astype(np.float32)))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class ImagesDataLoader(DataLoader):
    def __init__(self, data_path,
                 input_regex="gripper_force",
                 output_regex="strain_field_matrix_path"):
        self.cached_stamp_shape_matrix = None
        self.cached_stiffness_distributions_grippers_matrices = None
        self.preprocessed_data_path = None
        super().__init__(data_path, input_regex, output_regex)

    def create_dataset(self, x, y, batch_size, shuffle=True):
        x_scaled = self.scaler.scale(x)

        y_copy = y.copy(deep=True)
        y_copy['strain_field_matrix'] = y_copy['strain_field_matrix_path'].apply(
            lambda path: np.expand_dims(np.load(path), axis=-1)
        )
        y_scaled = self.scaler.scale(y_copy)
        y_scaled.drop(columns=['strain_field_matrix_path'], inplace=True)
        y_scaled_arrays = np.stack(y_scaled['strain_field_matrix'].values)

        dataset = tf.data.Dataset.from_tensor_slices((x_scaled.values.astype(np.float32), y_scaled_arrays))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class VectorImagesDataLoader(DataLoader):
    def __init__(self, data_path,
                 input_regex="gripper_force|stamp_shape_matrix_path",
                 output_regex="strain_field_matrix_path"):
        self.cached_stamp_shape_matrix = None
        self.cached_stiffness_distributions_grippers_matrices = None
        self.preprocessed_data_path = None
        super().__init__(data_path, input_regex, output_regex)

    def create_dataset(self, x, y, batch_size, shuffle=True):
        x_copy = x.copy(deep=True)
        x_copy['stamp_shape_matrix'] = x_copy['stamp_shape_matrix_path'].apply(
            lambda path: np.expand_dims(np.load(path), axis=-1)
        )
        x_scaled = self.scaler.scale(x_copy)
        x_scaled.drop(columns=['stamp_shape_matrix_path'], inplace=True)
        x_scaled_arrays = np.stack(x_scaled['stamp_shape_matrix'].values)
        x_scaled.drop(columns=['stamp_shape_matrix'], inplace=True)

        y_copy = y.copy(deep=True)
        y_copy['strain_field_matrix'] = y_copy['strain_field_matrix_path'].apply(
            lambda path: np.expand_dims(np.load(path), axis=-1)
        )
        y_scaled = self.scaler.scale(y_copy)
        y_scaled.drop(columns=['strain_field_matrix_path'], inplace=True)
        y_scaled_arrays = np.stack(y_scaled['strain_field_matrix'].values)

        x_vector_ds = tf.data.Dataset.from_tensor_slices(x_scaled.values.astype(np.float32))
        x_matrix_ds = tf.data.Dataset.from_tensor_slices(x_scaled_arrays)
        y_ds = tf.data.Dataset.from_tensor_slices(y_scaled_arrays)
        dataset = tf.data.Dataset.zip(((x_matrix_ds, x_vector_ds), y_ds))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
