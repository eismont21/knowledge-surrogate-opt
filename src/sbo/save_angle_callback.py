import numpy as np
import json
import os
from smac.callback import Callback
from src.scaler import Scaler
from src.constants import INPUT_SHAPE


class SaveAngleCallback(Callback):
    """
    Callback class that saves metrics during the SMAC optimization process.

    Attributes:
    - model (Model): An instance of the model to be used for optimization.
    - path (str): Path to the output directory.
    - metrics_dict (Dict[int, Dict[str, Union[float, int]]]): Dictionary storing metrics for each trial.
    - best_p_norm (float): Best observed p_norm value.
    - best_angle (Union[float, None]): Best observed angle corresponding to best_p_norm.
    - trial_counter (int): Keeps track of the number of trials.
    - scaler (Scaler): Instance of Scaler class for scaling operations.
    """

    def __init__(self, model, path: str) -> None:
        self.metrics_dict = {}
        self.best_p_norm = float('inf')
        self.best_angle = None
        self.trial_counter = 0
        self.scaler = Scaler()
        self.model = model
        self.path = path
        self._load_existing_data()
        super().__init__()

    def _load_existing_data(self) -> None:
        file_path = f'smac3_output/{self.path}/metrics.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                loaded_data = json.load(file)
                self.trial_counter = max(map(int, loaded_data.keys()))
                self.metrics_dict = {int(k): v for k, v in loaded_data.items()}
                self.best_p_norm = self.metrics_dict[self.trial_counter]['best_p_norm']
                self.best_angle = self.metrics_dict[self.trial_counter]['angle_for_best_p_norm']

    def on_tell_start(self, smbo, info, value) -> None:
        self.trial_counter += 1

        config = info.config

        x = []
        for i in range(INPUT_SHAPE[0]):
            x_i = config[f"{i:02}"]
            x.append(x_i)

        y = self.model.simulate(x).reshape(-1)
        current_angle = float(np.max(y))
        current_p_norm = float(value.cost)

        if current_p_norm < self.best_p_norm:
            self.best_p_norm = current_p_norm
            self.best_angle = current_angle

        self.metrics_dict[self.trial_counter] = {
            'angle': current_angle,
            'p_norm': current_p_norm,
            'best_p_norm': self.best_p_norm,
            'angle_for_best_p_norm': self.best_angle
        }

        with open(f'smac3_output/{self.path}/metrics.json', 'w') as file:
            json.dump(self.metrics_dict, file, indent=4)
