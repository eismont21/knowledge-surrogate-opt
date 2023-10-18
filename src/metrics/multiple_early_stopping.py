import numpy as np
import tensorflow as tf
from typing import List


class MultipleEarlyStopping(tf.keras.callbacks.Callback):
    """
    Callback to stop training the model based on multiple criteria.

    The MultipleEarlyStopping class allows monitoring multiple metrics and
    stops training when all these metrics have not improved for a specified
    number of epochs, defined by the `patience` parameter.

    Attributes:
    - monitors (List[str]): List of metric names to monitor.
    - modes (List[str]): List of 'min' or 'max' for each metric in monitors,
                         indicating whether a decrease or increase in the metric
                         value is considered better respectively.
    - patience (int): Number of epochs without improvement to wait before stopping training.
    - restore_best_weights (bool): Whether to restore model weights from the epoch with the
                                   best value of the monitored metric.
    - verbose (int): Level of verbosity. 0: silent, 1: print messages.
    - best_weights: Weights of the model from the best epoch.
    - wait (List[int]): Number of epochs since last improvement for each monitored metric.
    - stopped_epoch (int): The epoch at which training was stopped.
    - best (List[float]): Best value achieved for each monitored metric.
    """

    def __init__(self, monitors: List[str] = ['val_loss'], modes: List[str] = ['min'], patience: int = 0,
                 restore_best_weights: bool = True, verbose: int = 0) -> None:
        super(MultipleEarlyStopping, self).__init__()
        self.monitors = monitors
        self.modes = modes
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.verbose = verbose

        if len(monitors) != len(modes):
            raise ValueError('Number of monitors must be equal to number of modes.')

        for mode in modes:
            if mode not in ['min', 'max']:
                raise ValueError('Mode must be either "min" or "max".')

    def on_train_begin(self, logs=None) -> None:
        self.wait = [0] * len(self.monitors)
        self.stopped_epoch = 0
        self.best = [float('inf') if m == 'min' else -float('inf') for m in self.modes]

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        for i, monitor in enumerate(self.monitors):
            val_current = logs.get(monitor)
            if self.modes[i] == 'min':
                if np.less(val_current, self.best[i]):
                    self.best[i] = val_current
                    self.wait[i] = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait[i] += 1
            elif self.modes[i] == 'max':
                if np.greater(val_current, self.best[i]):
                    self.best[i] = val_current
                    self.wait[i] = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait[i] += 1

        if all(w > self.patience for w in self.wait):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print(f'\nRestoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None) -> None:
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f'\nEarly stopping occurred at epoch {self.stopped_epoch + 1} with best epoch {self.stopped_epoch + 1 - self.patience}')
