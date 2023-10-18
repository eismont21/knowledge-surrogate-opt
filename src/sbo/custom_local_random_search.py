from smac.acquisition.maximizer.local_and_random_search import LocalAndSortedRandomSearch
from ConfigSpace import Configuration


class CustomLocalAndSortedRandomSearch(LocalAndSortedRandomSearch):
    """
    CustomLocalAndSortedRandomSearch is an extension of the `LocalAndSortedRandomSearch` class. This custom implementation
    focuses on maximizing the acquisition function value by combining results from both random (5000 points)
    and local search methods, and then sorting the combined results based on the acquisition value.
    """

    def _maximize(
            self,
            previous_configs: list[Configuration],
            n_points: int,
    ) -> list[tuple[float, Configuration]]:
        next_configs_by_random_search_sorted = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=5000,
            _sorted=True,
        )[:10]

        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=next_configs_by_random_search_sorted,
        )

        next_configs_by_acq_value = next_configs_by_random_search_sorted + next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])

        return next_configs_by_acq_value
