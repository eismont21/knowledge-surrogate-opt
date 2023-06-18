import itertools
import json


def main():
    n_runs = 5
    train_sizes = [100, 250, 500, 750, 900]
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
                'epochs': 200,
                'early_stop_patience': 40,
            }
            configs.append(config)

    with open('configs_baseline.json', 'w') as f:
        json.dump(configs, f)

    models = ['UNet', 'EncoderDecoder', 'EncoderDecoderDropout']
    encodings = ['deepinsight', 'domain', 'domain_lengths', 'naive', 'multipath']

    parameters_product = itertools.product(train_sizes, models, encodings)

    configs = []

    for train_size, model, encoding in parameters_product:
        if encoding == 'multipath':
            data_type = 'vector_images'
        else:
            data_type = 'images'
        for i in range(n_runs):
            config = {
                'run': i,
                'data': {'type': data_type, 'train_size': train_size, 'test_size': 100},
                'model': model,
                'loss_function': 'mse',
                'optimizer': 'adam',
                'encoding': encoding,
                'positional_encoding': 0,
                'epochs': 200,
                'early_stop_patience': 40,
            }
            configs.append(config)

    with open('configs_cnn.json', 'w') as f:
        json.dump(configs, f)

if __name__ == "__main__":
    main()