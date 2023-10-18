"""
Script for generating and saving prediction metrics.

This script creates a series of K-Nearest Neighbor models, with varying values of 'k',
to predict matrices based on input data. Various metrics are used to evaluate the
quality of these predictions. Results for each 'k' and metric are saved to a CSV file.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import os
from src.metrics import RMSE, MAE, DifferenceObjectiveFunction, ToleranceAccuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

df = pd.read_csv('data/train_short.csv').filter(regex='gripper_force|strain_field_matrix_path')

matrices = []
for path in tqdm(df['strain_field_matrix_path']):
    matrix = np.load(path)
    matrices.append(matrix)

matrices = np.array(matrices)

df.drop(['strain_field_matrix_path'], axis=1, inplace=True)


def obj_function(matrix, p=4):
    return tf.norm(matrix, ord=p)


metrics_inverse = False

metrics = [RMSE(inverse=metrics_inverse),
           RMSE(name='rmse_max', obj_function=lambda x: tf.math.reduce_max(x), inverse=metrics_inverse),
           RMSE(name='rmse_obj', obj_function=obj_function, inverse=metrics_inverse),
           MAE(inverse=metrics_inverse),
           DifferenceObjectiveFunction(name='difference_max', obj_function=lambda x: tf.math.reduce_max(x),
                                       inverse=metrics_inverse),
           DifferenceObjectiveFunction(obj_function=obj_function, inverse=metrics_inverse),
           ToleranceAccuracy(tolerance=3, inverse=True)]

# Defining maximum 'k' for iteration
max_k = 25

k_metrics = {k: [] for k in range(1, max_k + 1)}
k_metrics_distance = {k: [] for k in range(1, max_k + 1)}

input_data = df.values

for k in tqdm(range(5, max_k + 1)):
    all_predicted_matrices, all_predicted_matrices_distance = [], []
    all_actual_matrices = []

    for i in range(len(df)):
        loo_input = np.delete(input_data, i, axis=0)
        loo_output = np.delete(matrices, i, axis=0)

        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        neigh.fit(loo_input)

        distances, indices = neigh.kneighbors([input_data[i]])
        weights = np.reciprocal(distances)
        weights = weights / np.sum(weights)

        predicted_matrix = tf.math.reduce_mean(tf.convert_to_tensor(loo_output[indices[0]]), axis=0)
        weighted_output = loo_output[indices[0]] * weights.reshape(-1, 1, 1)
        predicted_matrix_distance = weighted_output.sum(axis=0)

        all_predicted_matrices.append(tf.cast(predicted_matrix, tf.float32))
        all_predicted_matrices_distance.append(tf.cast(predicted_matrix_distance, tf.float32))
        all_actual_matrices.append(tf.cast(matrices[i], tf.float32))

    all_predicted_matrices = tf.stack(all_predicted_matrices)
    all_predicted_matrices_distance = tf.stack(all_predicted_matrices_distance)
    all_actual_matrices = tf.stack(all_actual_matrices)

    for metric in metrics:
        metric.reset_states()
        metric_value = metric(all_actual_matrices, all_predicted_matrices).numpy()
        k_metrics[k].append(metric_value)
        metric.reset_states()

        metric_value = metric(all_actual_matrices, all_predicted_matrices_distance).numpy()
        k_metrics_distance[k].append(metric_value)
        metric.reset_states()

metric_names = [metric.name for metric in metrics]
result_list = []
for mode, data in [('no_distance', k_metrics), ('distance', k_metrics_distance)]:
    for k, values in data.items():
        result_dict = {'k': k, 'mode': mode}
        result_dict.update({metrics[idx].name: val for idx, val in enumerate(values)})
        result_list.append(result_dict)

df_results = pd.DataFrame(result_list)

df_results.to_csv(f'results_knn_{max_k}.csv', index=False)
