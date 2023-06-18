import collections
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import time
from multiprocessing import Pool
from tqdm import tqdm
from src.models import DenseModel, DenseModelDropout, UNet, MultiPathUNet, EncoderDecoder, \
    MultiPathEncoderDecoder
from src.dataloaders import BaselineDataLoader, ImagesDataLoader, VectorImagesDataLoader
from src.loss_functions import SSIMLoss, TotalLoss, WeightedLoss
from src.metrics import SSIMLossMetric, TotalLossMetric, WeightedLossMetric, MAE, RMSE


def iterative_split(iterations, train_size=None, test_size=None, train_ratio=None,
                    test_ratio=None, seed=42):
    data_loader = BaselineDataLoader('data/train_short.csv')
    x, _ = data_loader.load_data()

    np.random.seed(seed)

    freq_map_train = collections.defaultdict(int)
    freq_map_test = collections.defaultdict(int)
    indices_all = list(x.index)
    np.random.shuffle(indices_all)

    splits = []

    for _ in range(iterations):
        indices_train_sorted = sorted(indices_all, key=lambda x: freq_map_train[x])
        train_indices = indices_train_sorted[:train_size]
        for idx in train_indices:
            freq_map_train[idx] += 1

        indices_test_sorted = sorted([i for i in indices_all if i not in train_indices], key=lambda x: freq_map_test[x])
        test_indices = indices_test_sorted[:test_size]
        for idx in test_indices:
            freq_map_test[idx] += 1

        splits.append((train_indices, test_indices))

    return splits


def get_data_loader(config):
    if config['data']['type'] == 'vector':
        data_loader = BaselineDataLoader('data/train_short.csv')
    elif config['data']['type'] == 'images':
        data_loader = ImagesDataLoader('data/train_short.csv')
    elif config['data']['type'] == 'vector_images':
        data_loader = VectorImagesDataLoader('data/train_short.csv')
    else:
        raise ValueError(f"Unknown data loader type: {config['data']['type']}")

    return data_loader


def get_model(config, train_dataset, x_train):
    if config['model'] == 'Baseline':
        model = DenseModel(name='Baseline',
                           input_dim=train_dataset.element_spec[0].shape[1:],
                           output_dim=train_dataset.element_spec[1].shape[1:])
    elif config['model'] == 'BaselineDropout':
        model = DenseModelDropout(name='BaselineDropout',
                                  input_dim=train_dataset.element_spec[0].shape[1:],
                                  output_dim=train_dataset.element_spec[1].shape[1:],
                                  x_train=x_train)
    elif config['model'] == 'UNet':
        if config['encoding'] == 'multipath':
            model = MultiPathUNet(name='MultiPathUNet',
                                  input_dim=(
                                      train_dataset.element_spec[0][0].shape[1:],
                                      train_dataset.element_spec[0][1].shape[1:]),
                                  output_dim=train_dataset.element_spec[1].shape[1:],
                                  positional_encoding=config['positional_encoding'],
                                  x_train=x_train)
        else:
            model = UNet(name='UNet',
                         input_dim=train_dataset.element_spec[0].shape[1:],
                         output_dim=train_dataset.element_spec[1].shape[1:],
                         encoding=config['encoding'],
                         positional_encoding=config['positional_encoding'],
                         x_train=x_train)
    elif config['model']['type'] == 'EncoderDecoder':
        if config['encoding'] == 'multipath':
            model = MultiPathEncoderDecoder(name='MultiPathEncoderDecoder',
                                            input_dim=(
                                                train_dataset.element_spec[0][0].shape[1:],
                                                train_dataset.element_spec[0][1].shape[1:]),
                                            output_dim=train_dataset.element_spec[1].shape[1:],
                                            positional_encoding=config['positional_encoding'],
                                            x_train=x_train)
        else:
            model = EncoderDecoder(name='EncoderDecoder',
                                   input_dim=train_dataset.element_spec[0].shape[1:],
                                   output_dim=train_dataset.element_spec[1].shape[1:],
                                   encoding=config['encoding'],
                                   positional_encoding=config['positional_encoding'],
                                   x_train=x_train)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    return model


def get_loss_function(config):
    if config['loss_function'] == 'mse':
        loss_function = 'mse'
        loss_metric = RMSE(name='Loss_MSE', squared=True, inverse=True)
    elif config['loss_function'] == 'mae':
        loss_function = 'mae'
        loss_metric = MAE(name='Loss_MAE', inverse=True)
    elif config['loss_function'] == 'ssim':
        loss_function = SSIMLoss()
        loss_metric = SSIMLossMetric(name='Loss_SSIM')
    elif config['loss_function'] == 'weighted_mse':
        loss_function = WeightedLoss(obj_function=p_norm, loss_fn=tf.keras.losses.MeanSquaredError())
        loss_metric = WeightedLossMetric(name='Loss_Weighted_MSE', obj_function=p_norm,
                                         loss_fn=tf.keras.losses.MeanSquaredError(), inverse=True)
    elif config['loss_function'] == 'weighted_mae':
        loss_function = WeightedLoss(obj_function=p_norm, loss_fn=tf.keras.losses.MeanAbsoluteError())
        loss_metric = WeightedLossMetric(name='Loss_Weighted_MAE', obj_function=p_norm,
                                         loss_fn=tf.keras.losses.MeanAbsoluteError(), inverse=True)
    elif config['loss_function'] == 'total_mse':
        loss_function = TotalLoss(obj_function=p_norm, loss_fn=tf.keras.losses.MeanSquaredError())
        loss_metric = TotalLossMetric(name='Loss_Total_MSE', obj_function=p_norm,
                                      loss_fn=tf.keras.losses.MeanSquaredError())
    elif config['loss_function'] == 'total_mae':
        loss_function = TotalLoss(obj_function=p_norm, loss_fn=tf.keras.losses.MeanAbsoluteError())
        loss_metric = TotalLossMetric(name='Loss_Total_MAE', obj_function=p_norm,
                                      loss_fn=tf.keras.losses.MeanAbsoluteError())
    elif config['loss_function'] == 'total_ssim':
        loss_function = TotalLoss(obj_function=p_norm, loss_fn=SSIMLoss())
        loss_metric = TotalLossMetric(name='Loss_Total_SSIM', obj_function=p_norm, loss_fn=SSIMLoss())
    else:
        raise ValueError(f"Unknown loss function: {config['loss_function']}")

    return loss_function, loss_metric


def save_history(history, config):
    os.makedirs(os.path.join(config['output_dir'], 'metrics'), exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv(os.path.join(config['output_dir'], 'metrics', 'history.csv'), index=False)
    return history_df


def save_history_plots(history_df, config):
    metrics_dir = os.path.join(config['output_dir'], 'metrics')
    metrics = [col for col in history_df.columns if not col.startswith('val_')]
    dropout_metrics = [col for col in metrics if 'dropout' in col]
    other_metrics = [col for col in metrics if col not in dropout_metrics]

    for metric in other_metrics:
        name = metric.replace("_", " ")
        plt.figure(figsize=(10, 10), dpi=300)
        sns.lineplot(data=history_df, x='epoch', y=metric, label='Train')
        sns.lineplot(data=history_df, x='epoch', y=f'val_{metric}', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name.title())
        plt.legend()
        plt.savefig(os.path.join(metrics_dir, f'{metric.lower()}.png'))
        plt.close()

    if len(dropout_metrics) != 0:
        plt.figure(figsize=(10, 10), dpi=300)
        sns.lineplot(data=history_df, x='epoch', y=dropout_metrics)
        plt.xlabel('Epoch')
        plt.ylabel('Dropout Rate')
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [label.replace('dropout_rate_', '') for label in labels]
        plt.legend(handles, new_labels)
        plt.title('Concrete Dropout')
        plt.savefig(os.path.join(metrics_dir, 'dropouts.png'))
        plt.close()


def save_result(result, name, config):
    with open(os.path.join(config['output_dir'], f'{name}.json'), 'w') as f:
        json.dump(result, f)


def save_config(config):
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f)


def run_experiment(config):
    data_loader = get_data_loader(config)
    x, y = data_loader.load_data()
    train_indices = config['data']['train_indices']
    test_indices = config['data']['test_indices']
    x_train, x_test = x.loc[train_indices], x.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    train_dataset = data_loader.create_dataset(x_train, y_train, batch_size=8)
    val_dataset = data_loader.create_dataset(x_test, y_test, batch_size=8, shuffle=False)

    model = get_model(config, train_dataset, x_train)
    loss_function, loss_metric = get_loss_function(config)
    optimizer = config['optimizer']
    model.compile(optimizer=optimizer, loss=loss_function, loss_metric=loss_metric, obj_function=p_norm)
    history = model.fit(train_dataset, val_dataset,
                        epochs=config['epochs'], early_stop_patience=config['early_stop_patience'], verbose=0,
                        best_model_filepath=config['output_dir'])
    history_df = save_history(history, config)
    save_history_plots(history_df, config)

    result = model.evaluate(val_dataset, verbose=0)
    save_result(result, 'result', config)

    if config['model'] != 'Baseline':
        model.reload(is_mc_dropout=True)
        runs = [5, 25, 100]
        for run in runs:
            result = model.mc_evaluate(val_dataset, run)
            save_result(result, f'result_{run}', config)


def worker(config_index):
    gpu_id = config_index % 4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    config = configs[config_index]
    while True:
        dir_name = time.strftime('%d-%m-%Y_%H-%M-%S')
        config['output_dir'] = os.path.join('experiments', dir_name)
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
            break
        else:
            time.sleep(random.randint(1, 10))
    config['data']['train_indices'] = splits[config['data']['train_size']][config['run']][0]
    config['data']['test_indices'] = splits[config['data']['train_size']][config['run']][1]
    save_config(config)

    run_experiment(config)


def p_norm(matrix, p=4):
    return tf.norm(matrix, ord=p)


with open('configs_baseline.json', 'r') as f:
    configs = json.load(f)

n_runs = 5

splits = {
    100: iterative_split(n_runs, train_size=100, test_size=100, random_state=42),
    250: iterative_split(n_runs, train_size=250, test_size=100, random_state=43),
    500: iterative_split(n_runs, train_size=500, test_size=100, random_state=44),
    750: iterative_split(n_runs, train_size=750, test_size=100, random_state=45),
    900: iterative_split(n_runs, train_size=900, test_size=100, random_state=46)
}


def main():
    with Pool(4) as p:
        for _ in tqdm(p.imap(worker, range(len(configs))), total=len(configs)):
            pass


if __name__ == "__main__":
    main()
