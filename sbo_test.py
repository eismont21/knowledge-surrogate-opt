import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from smac import Scenario
from smac import BlackBoxFacade as BBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.function.confidence_bound import LCB
from smac.acquisition.maximizer.local_and_random_search import LocalAndSortedRandomSearch
from smac.acquisition.maximizer.differential_evolution import DifferentialEvolution
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.intensifier.intensifier import Intensifier
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.main.config_selector import ConfigSelector
from ConfigSpace import ConfigurationSpace, Float, Configuration
from src.sbo import Oracle, SaveAngleCallback, SurrogateModel
from src.models import DenseModelDropout, CFPNetM
from src.dataloaders import BaselineDataLoader, ImagesDataLoader
from src.constants import INPUT_SHAPE, OUTPUT_SHAPE
import tensorflow as tf
import math
from tqdm import tqdm

import numpy as np
from scipy.stats import norm
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

oracle = Oracle()
seed = 0
ex_name = 'test_cfpnetm'


def p_norm(matrix, p=4):
    return float(tf.norm(matrix, ord=p))


cs = ConfigurationSpace(seed=seed)
for i in range(INPUT_SHAPE[0]):
    x = Float(f"{i:02}", (0.01, 1))
    cs.add_hyperparameters([x])

scenario = Scenario(cs,
                    name=ex_name,
                    deterministic=True,
                    n_trials=150)


def obj_function(config: Configuration, seed: int = seed) -> float:
    x = []
    for i in range(INPUT_SHAPE[0]):
        x_i = config[f"{i:02}"]
        x.append(x_i)
    y = oracle.simulate(x)

    return p_norm(y)


# model = DenseModelDropout(name='BaselineDropout',
#                          input_dim=INPUT_SHAPE,
#                          output_dim=(math.prod(OUTPUT_SHAPE),),
#                          is_mc_dropout=True)
# data_loader = BaselineDataLoader('data/val_short.csv')

model = CFPNetM(name='CFPNetM',
                input_dim=INPUT_SHAPE,
                output_dim=OUTPUT_SHAPE,
                encoding="domain",
                positional_encoding=1,
                is_mc_dropout=True)
data_loader = ImagesDataLoader('data/val_short.csv')
sur_model = SurrogateModel(cs, model, data_loader, p_norm, oracle)


# pbar = tqdm(desc="Optimizing", unit="config")


class CustomEI(EI):
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute EI acquisition value

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.

        Raises
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EI is < 0 for at least one sample (normal function value space).
        ValueError
            If EI is < 0 for at least one sample (log function value space).
        """
        assert self._model is not None
        assert self._xi is not None

        pbar.update(X.shape[0])

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)
            # print("m: ", m.reshape(-1))
            # print("v: ", v.reshape(-1))
            s = np.sqrt(v)

            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                return (self._eta - m - self._xi) * norm.cdf(z) + s * norm.pdf(z)

            if np.any(s == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                # logger.warning("Predicted std is 0.0 for at least one sample.")
                s_copy = np.copy(s)
                s[s_copy == 0.0] = 1.0
                f = calculate_f()
                f[s_copy == 0.0] = 0.0
            else:
                f = calculate_f()

            if (f < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

            return f
        else:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, var_ = self._model.predict_marginalized(X)
            std = np.sqrt(var_)

            def calculate_log_ei() -> np.ndarray:
                # we expect that f_min is in log-space
                assert self._eta is not None
                assert self._xi is not None

                f_min = self._eta - self._xi
                v = (f_min - m) / std
                return (np.exp(f_min) * norm.cdf(v)) - (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

            if np.any(std == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                # logger.warning("Predicted std is 0.0 for at least one sample.")
                std_copy = np.copy(std)
                std[std_copy == 0.0] = 1.0
                log_ei = calculate_log_ei()
                log_ei[std_copy == 0.0] = 0.0
            else:
                log_ei = calculate_log_ei()

            if (log_ei < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one sample.")

            return log_ei.reshape((-1, 1))


class CustomDifferentialEvolution(DifferentialEvolution):
    def _maximize(
            self,
            previous_configs: list[Configuration],
            n_points: int,
    ) -> list[tuple[float, Configuration]]:
        configs: list[tuple[float, Configuration]] = []

        def func(x: np.ndarray) -> np.ndarray:
            assert self._acquisition_function is not None
            return -self._acquisition_function([Configuration(self._configspace, vector=x)])

        ds = DifferentialEvolutionSolver(
            func,
            bounds=[[0, 1] for _ in range(len(self._configspace))],
            args=(),
            strategy="best1bin",
            maxiter=25,
            popsize=50,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self._rng.randint(1000),
            polish=True,
            callback=None,
            disp=False,
            init="latinhypercube",
            atol=0,
        )

        _ = ds.solve()
        for pop, val in zip(ds.population, ds.population_energies):
            rc = Configuration(self._configspace, vector=pop)
            rc.origin = "Acquisition Function Maximizer: Differential Evolution"
            configs.append((-val, rc))

        configs.sort(key=lambda t: t[0])
        configs.reverse()

        return configs


class CustomLocalAndSortedRandomSearch(LocalAndSortedRandomSearch):
    def _maximize(
            self,
            previous_configs: list[Configuration],
            n_points: int,
    ) -> list[tuple[float, Configuration]]:
        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=1000,
            _sorted=True,
        )

        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=next_configs_by_random_search_sorted,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = next_configs_by_random_search_sorted + next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        # logger.debug(f"First 5 acquisition function values of selected configurations:\n{', '.join(first_five)}")

        return next_configs_by_acq_value


smac = HPOFacade(
    scenario=scenario,
    target_function=obj_function,
    model=sur_model,
    acquisition_function=EI(),
    # acquisition_maximizer=CustomDifferentialEvolution(
    #    configspace=scenario.configspace,
    #    challengers=1000,
    #    seed=scenario.seed,
    # ),
    acquisition_maximizer=CustomLocalAndSortedRandomSearch(
        configspace=scenario.configspace,
        challengers=5000,
        local_search_iterations=1,
        # n_steps_plateau_walk=2,
        # max_steps=2,
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
    ),
    runhistory_encoder=RunHistoryEncoder(scenario),
    config_selector=ConfigSelector(scenario, retrain_after=10),
    overwrite=True,
    logging_level=20,
    callbacks=[SaveAngleCallback(oracle, f'{ex_name}/{seed}')]
)

incumbent = smac.optimize()

# pbar.close()

# Get cost of default configuration
default_cost = smac.validate(cs.get_default_configuration())
print(f"Default cost: {default_cost}")

# Let's calculate the cost of the incumbent
incumbent_cost = smac.validate(incumbent)
print(f"Incumbent cost: {incumbent_cost}")
