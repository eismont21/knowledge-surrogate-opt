import numpy as np
import tensorflow as tf


class MultipleEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitors=['val_loss'], modes=['min'], patience=0,
                 restore_best_weights=True, verbose=0):
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

    def on_train_begin(self, logs=None):
        self.wait = [0] * len(self.monitors)
        self.stopped_epoch = 0
        self.best = [float('inf') if m == 'min' else -float('inf') for m in self.modes]

    def on_epoch_end(self, epoch, logs=None):
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

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f'\nEarly stopping occurred at epoch {self.stopped_epoch + 1} with best epoch {self.stopped_epoch + 1 - self.patience}')
