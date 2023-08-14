import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import gc
from datetime import datetime
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger
from src.loss_functions import WeightedLoss
from src.metrics import WeightedLossMetric
from src.scaler import Scaler
from src.constants import MATRIX_SHAPE

logger = get_logger(__name__)


class SurrogateModel(AbstractModel):
    def __init__(self, configspace, model, data_loader, obj_function, oracle, n_inferences: int = 40,
                 chunk_size: int = 100):
        super().__init__(configspace)
        self.oracle = oracle
        self.scaler = Scaler()
        self.model = model
        self.data_loader = data_loader
        x, y = self.data_loader.load_data()
        self.x_columns = x.columns
        self.y_columns = y.columns
        self.val_dataset = self.data_loader.create_dataset(x, y, batch_size=8, shuffle=False)

        self.n_inferences = n_inferences
        self.chunk_size = chunk_size
        self.obj_function = obj_function
        self.save_filepath = 'test_sbo'
        self.opt_start = None
        self.pbar = None

        optimizer = 'adam'
        loss_function = WeightedLoss(obj_function=self.obj_function, loss_fn=tf.keras.losses.MeanAbsoluteError())
        loss_metric = WeightedLossMetric(name='Loss_Weighted_MAE', obj_function=self.obj_function,
                                         loss_fn=tf.keras.losses.MeanAbsoluteError(), inverse=True)
        self.model.compile(optimizer=optimizer, loss=loss_function, loss_metric=loss_metric,
                           obj_function=self.obj_function, lr=0.001)

        os.makedirs(os.path.join(self.save_filepath, 'strain_fields'), exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray):
        gc.collect()
        X = self.scaler.inverse_transform(X, col_name="gripper_force")
        if self.opt_start is not None:
            self.pbar.close()
            logger.info(f"Optimization took around {datetime.now() - self.opt_start}")
        logger.info(f"Training starts with {X.shape[0]} samples")
        start_train = datetime.now()

        x_train = pd.DataFrame(X, columns=self.x_columns)
        y_values = [self._create_strain_field(index, row) for index, row in x_train.iterrows()]
        y_train = pd.DataFrame(y_values, columns=self.y_columns)
        train_dataset = self.data_loader.create_dataset(x_train, y_train, batch_size=8)

        self._reset_weights()
        self.model.set_train_size(X.shape[0])
        _ = self.model.train(train_dataset, self.val_dataset,
                             epochs=5, early_stop_patience=100, verbose=0,
                             save_filepath=self.save_filepath + f'/{X.shape[0]}',
                             save_history=True, is_sbo=True)
        _ = self.model.evaluate(self.val_dataset, verbose=0, save_result=True)
        dur_train = datetime.now() - start_train
        logger.info(f"Training took {dur_train}")
        self.opt_start = datetime.now()
        self.pbar = tqdm(desc="Optimizing", unit="config")

        return self

    def predict(self, x: np.ndarray, covariance_type: str | None = "diagonal"):
        final_mean = np.zeros(x.shape[0])
        final_variance = np.zeros(x.shape[0])

        for start_idx in range(0, x.shape[0], self.chunk_size):
            x_chunk = x[start_idx:start_idx + self.chunk_size]

            # Repeat the input data for all inferences
            repeated_x = np.repeat(x_chunk, self.n_inferences, axis=0)

            # Predict in a single call
            chunk_predictions = self.model.predict(repeated_x)

            # Transform predictions
            transformed_predictions = self.scaler.inverse_transform(
                chunk_predictions, col_name="strain_field_matrix"
            ).reshape(-1, self.n_inferences, *MATRIX_SHAPE)

            # Apply objective function
            results = np.array([[self.obj_function(pred) for pred in batch] for batch in transformed_predictions])

            mean = results.mean(axis=1)
            variance = results.var(axis=1)

            # Update the pre-allocated arrays
            final_mean[start_idx:start_idx + len(mean)] = mean
            final_variance[start_idx:start_idx + len(variance)] = variance

            self.pbar.update(len(mean))

        final_mean = final_mean.reshape(-1, 1)
        final_variance = final_variance.reshape(-1, 1)

        logger.debug(
            f"Minimal predicted cost: {round(np.min(final_mean), 4)} ({round(final_variance[np.argmin(final_mean)][0], 2)})")

        return final_mean, final_variance

    def _create_strain_field(self, index, x):
        x = x.to_numpy()
        strain_field = self.oracle.simulate(x)
        path = os.path.join(self.save_filepath, 'strain_fields', f'{index}.npy')
        np.save(path, strain_field)
        return str(path)

    def predict_marginalized_over_instances(self, X: np.ndarray):
        return self.predict(X)

    def _reset_weights(self):
        for layer in self.model.model.layers:
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                var = getattr(layer, k.replace("_initializer", ""))
                init_class = type(initializer)
                new_initializer = init_class()
                var.assign(new_initializer(var.shape, var.dtype))
