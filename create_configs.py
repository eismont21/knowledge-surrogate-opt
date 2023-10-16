import itertools
import json
import random


def main():
    n_runs = 5
    #train_sizes = [100, 250, 500, 750, 900]
    train_sizes = [100, 250, 500, 900]
    #train_sizes = [250]
    models = ['Baseline', 'BaselineDropout']
    loss_functions = ['mse', 'mae', 'ssim', 'weighted_mse', 'weighted_mae', 'total_mse', 'total_mae', 'total_ssim']

    parameters_product = itertools.product(train_sizes, models, loss_functions)

    configs = []

    for train_size, model, loss_function in parameters_product:
        for i in range(n_runs):
            config = {
                'run': i,
                'data': {'type': 'vector', 'train_size': train_size, 'test_size': 100},
                'model': model,
                'loss_function': loss_function,
                'optimizer': 'adam',
                'encoding': 'none',
                'positional_encoding': 0,
                'epochs': 300,
                'early_stop_patience': 60,
            }
            #configs.append(config)

    models = ['UNet', 'CFPNetM']
    encodings = ['domain']
    pos_encodings = [1, 2]
    loss_functions = ['mae', 'ssim', 'weighted_mse', 'weighted_mae', 'total_mse', 'total_mae', 'total_ssim']

    parameters_product = itertools.product(train_sizes, models, loss_functions)

    for train_size, model, loss_function in parameters_product:
        for i in range(n_runs):
            config = {
                'run': i,
                'data': {'type': 'images', 'train_size': train_size, 'test_size': 100},
                'model': model,
                'loss_function': loss_function,
                'optimizer': 'adam',
                'encoding': 'domain',
                'positional_encoding': 0,
                'epochs': 300,
                'early_stop_patience': 60,
            }
            configs.append(config)

    random.shuffle(configs)
    configs = sorted(configs, key=lambda x: x['run'])
    with open('configs_loss.json', 'w') as f:
        json.dump(configs, f)


if __name__ == "__main__":
    main()
