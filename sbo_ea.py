"""
Script for Hyperparameter Optimization using DEAP.

This script initializes a genetic algorithm with specific configurations and uses
the DEAP framework to perform genetic operations like mutation, crossover, and selection.
During the algorithm's execution, it simulates and evaluates individuals' performance
using an Oracle instance, tracking and recording the best results. The script also
keeps a record of the algorithm's progression, storing each trial's details and
finally saving this data as a CSV file for further analysis.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tqdm import tqdm
from src.sbo import Oracle

FOLDER = 'smac3_output/ea/'

seed = 0

folder_run = os.path.join(FOLDER, str(seed))
os.makedirs(folder_run, exist_ok=True)

oracle = Oracle()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

BOUND_LOW, BOUND_UP = 0.01, 1.0  # adjusting the range
NDIM = 60  # dimensions

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def p_norm(matrix, p=4):
    """
    The user-defined objective function.
    """
    return float(tf.norm(matrix, ord=p))


def eval(individual):
    y = oracle.simulate(individual)
    return p_norm(y)


def get_max_angle(individual):
    y = oracle.simulate(individual)
    return np.max(y)


toolbox.register("evaluate", eval)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=0.08)
toolbox.register("select", tools.selTournament, tournsize=3)

history = []

n_trials = 1000

eval_count = 0
best_p_norm = float('inf')
best_angle = None

random.seed(seed)

# creating population
pop = toolbox.population(n=14)

pbar = tqdm(total=n_trials)

# Algorithm loop
while eval_count < n_trials:
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.08)
    fits = toolbox.map(toolbox.evaluate, offspring)

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = (fit,)

        trial_data = {'id': eval_count + 1}
        trial_data.update({f'{i:02}': ind[i] for i in range(len(ind))})
        current_angle = get_max_angle(ind)
        current_p_norm = fit
        trial_data['angle'] = current_angle
        trial_data['p_norm'] = current_p_norm
        if current_p_norm < best_p_norm:
            best_p_norm = current_p_norm
            best_angle = current_angle
        trial_data['best_p_norm'] = best_p_norm
        trial_data['angle_for_best_p_norm'] = best_angle
        history.append(trial_data)
        pbar.update(1)
        eval_count += 1
        if eval_count >= n_trials:
            break

    pop = toolbox.select(offspring, k=len(pop))

pbar.close()

df = pd.DataFrame(history)
df.to_csv(os.path.join(folder_run, 'results.csv'), index=False)
