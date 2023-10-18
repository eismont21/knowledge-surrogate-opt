"""
Script for Hyperparameter Optimization using SMAC and CFPNet-M as surrogate model.

This script initializes a hyperparameter configuration space, sets up a specific scenario for optimization,
defines the objective function to be minimized, and then uses the SMAC framework to perform optimization on
the given objective. After optimization, it validates and prints the results for both the default configuration
and the found incumbent configuration.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.acquisition.function.expected_improvement import EI
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.intensifier.intensifier import Intensifier
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.main.config_selector import ConfigSelector
from ConfigSpace import ConfigurationSpace, Float, Configuration
from src.sbo import Oracle, SaveAngleCallback, SurrogateModel, CustomLocalAndSortedRandomSearch
from src.models import DenseModelDropout, CFPNetM
from src.dataloaders import ImagesDataLoader
from src.constants import INPUT_SHAPE, OUTPUT_SHAPE
import tensorflow as tf

oracle = Oracle()
seed = 0
ex_name = 'cfpnetm'


def p_norm(matrix, p=4):
    """
    The user-defined objective function.
    """
    return float(tf.norm(matrix, ord=p))


cs = ConfigurationSpace(seed=seed)
for i in range(INPUT_SHAPE[0]):
    x = Float(f"{i:02}", (0.01, 1))
    cs.add_hyperparameters([x])

scenario = Scenario(cs,
                    name=ex_name,
                    seed=seed,
                    deterministic=True,
                    n_trials=1000)


def obj_function(config: Configuration, seed: int = seed) -> float:
    x = []
    for i in range(INPUT_SHAPE[0]):
        x_i = config[f"{i:02}"]
        x.append(x_i)
    y = oracle.simulate(x)

    return p_norm(y)


model = CFPNetM(name='CFPNetM',
                input_dim=INPUT_SHAPE,
                output_dim=OUTPUT_SHAPE,
                encoding="domain",
                positional_encoding=1,
                is_mc_dropout=True)
data_loader = ImagesDataLoader('data/val_short.csv')
sur_model = SurrogateModel(cs, model, data_loader, p_norm, oracle, save_filepath=f"smac3_output/{ex_name}/{seed}")
run_history_encoder = RunHistoryEncoder(scenario)
if os.path.exists(f"smac3_output/{ex_name}/{seed}/runhistory.json"):
    run_history = RunHistory()
    run_history.load(f"smac3_output/{ex_name}/{seed}/runhistory.json", cs)
    run_history_encoder.runhistory = run_history

smac = HPOFacade(
    scenario=scenario,
    target_function=obj_function,
    model=sur_model,
    acquisition_function=EI(),
    acquisition_maximizer=CustomLocalAndSortedRandomSearch(
        configspace=scenario.configspace,
        challengers=1000,
        local_search_iterations=1,
        seed=scenario.seed,
    ),
    initial_design=SobolInitialDesign(
        scenario=scenario,
        n_configs=100,
        max_ratio=1,
        seed=scenario.seed,
    ),
    random_design=ProbabilityRandomDesign(seed=scenario.seed, probability=0.08447232371720552),
    intensifier=Intensifier(
        scenario=scenario,
        max_config_calls=3,
        max_incumbents=20,
        retries=1000,
    ),
    runhistory_encoder=run_history_encoder,
    config_selector=ConfigSelector(scenario, retrain_after=10),
    overwrite=False,
    logging_level=20,
    callbacks=[SaveAngleCallback(oracle, f'{ex_name}/{seed}')]
)

incumbent = smac.optimize()

# Get cost of default configuration
default_cost = smac.validate(cs.get_default_configuration())
print(f"Default cost: {default_cost}")

# Let's calculate the cost of the incumbent
incumbent_cost = smac.validate(incumbent)
print(f"Incumbent cost: {incumbent_cost}")
