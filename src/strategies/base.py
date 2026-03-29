from abc import ABC, abstractmethod

import pandas as pd


class TrajectorySelectionMethod(ABC):
    """Abstract base class for trajectory selection methods.

    All specific selection strategies must inherit from this class and
    implement the `select` method.
    """

    def __init__(self, proportion: float, seed: int = 1415) -> None:
        """Args:
        proportion: The fraction of trajectories to select (0.0 to 1.0).
        seed: Random seed for reproducibility.
        """
        self.proportion = proportion
        self.seed = seed

    @abstractmethod
    def select(self, df: pd.DataFrame) -> list[str]:
        """Selects a subset of trajectories from the dataframe.

        Args:
            df: DataFrame containing trajectory features. Must contain 'tid' and 'label'.

        Returns:
            List of selected trajectory IDs (strings).
        """
        pass
