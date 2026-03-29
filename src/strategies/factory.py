from src.strategies.base import TrajectorySelectionMethod
from src.strategies.selectors import (
    RandomTrajectorySelection,
    SelectTrajectoryBasedOnDiversityMaximization,
    SelectTrajectoryBasedOnOutlierness,
    SelectTrajectoryBasedOnRepresentativeness,
    SelectTrajectoryBasedOnUncertainty,
)


def get_strategy_class(name: str) -> type[TrajectorySelectionMethod]:
    """Factory function to retrieve the strategy class by name.

    Args:
        name: The name of the strategy ('random', 'outlierness', etc.)

    Returns:
        The class (not instantiated).
    """
    mapping = {
        "random": RandomTrajectorySelection,
        "outlierness": SelectTrajectoryBasedOnOutlierness,
        "diversity": SelectTrajectoryBasedOnDiversityMaximization,
        "representativeness": SelectTrajectoryBasedOnRepresentativeness,
        "uncertainty": SelectTrajectoryBasedOnUncertainty,
    }

    if name.lower() not in mapping:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(mapping.keys())}")

    return mapping[name.lower()]
