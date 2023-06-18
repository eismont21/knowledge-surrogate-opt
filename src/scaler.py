import pandas as pd


class Scaler:
    def __init__(self):
        self.COL_MIN_MAX = {
            "gripper_x": [0, 300],
            "gripper_y": [0, 460],
            "gripper_dir_x": [-10, 10],
            "gripper_dir_y": [-10, 10],
            "gripper_force": [0.01, 1],
            "characteristic_e_1": [5, 200],
            "characteristic_e_2": [5, 20],
            "characteristic_nu_12": [0, 0.5],
            "characteristic_g_12": [0, 0.5],
            "stiffness_q_11": [5, 200],
            "stiffness_q_12": [0, 20],
            "stiffness_q_22": [5, 20],
            "stiffness_q_33": [0, 0.5],
            "length": [0, 60],
            "angle": [0, 90],
            "strain_field_matrix": [0, 90],
            "stamp_shape_matrix": [0, 100]
        }

    def scale(self, data, col_name=None):
        if isinstance(data, pd.DataFrame):
            scaled_data = data.copy(deep=True)
            for pattern, (min_val, max_val) in self.COL_MIN_MAX.items():
                matching_cols = [col for col in data.columns if pattern in col and "path" not in col]
                if matching_cols:
                    scaled_data[matching_cols] = self._scale_columns(scaled_data[matching_cols], min_val, max_val)
            return scaled_data
        else:
            min_val, max_val = self.COL_MIN_MAX[col_name]
            return self._scale_columns(data, min_val, max_val)

    @staticmethod
    def _scale_columns(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

    def inverse_transform(self, data, col_name=None):
        if isinstance(data, pd.DataFrame):
            scaled_df = data.copy(deep=True)
            for pattern, (min_val, max_val) in self.COL_MIN_MAX.items():
                matching_cols = [col for col in data.columns if pattern in col]
                if matching_cols:
                    scaled_df[matching_cols] = self._inverse_transform_columns(scaled_df[matching_cols], min_val,
                                                                               max_val)
            return scaled_df
        else:
            min_val, max_val = self.COL_MIN_MAX[col_name]
            return self._inverse_transform_columns(data, min_val, max_val)

    @staticmethod
    def _inverse_transform_columns(data, min_val, max_val):
        return data * (max_val - min_val) + min_val
