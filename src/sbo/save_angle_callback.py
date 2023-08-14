import numpy as np
import json
from smac.callback import Callback
from src.scaler import Scaler
from src.constants import INPUT_SHAPE

class SaveAngleCallback(Callback):
    def __init__(self, model, path):
        self.metrics_dict = {}
        self.trial_counter = 0
        self.scaler = Scaler()
        self.model = model
        self.path = path
        super().__init__()

    def on_tell_start(self, smbo, info, value):
        self.trial_counter += 1

        config = info.config

        x = []
        for i in range(INPUT_SHAPE[0]):
            x_i = config[f"{i:02}"]
            x.append(x_i)

        y = self.model.simulate(x).reshape(-1)

        self.metrics_dict[self.trial_counter] = {
            'angle': float(np.max(y)),
            'p_norm': float(value.cost),
        }

        with open(f'smac3_output/{self.path}/metrics.json', 'w') as file:
            json.dump(self.metrics_dict, file, indent=4)