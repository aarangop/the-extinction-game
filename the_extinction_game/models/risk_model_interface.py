from abc import ABC, abstractmethod
import numpy as np


class RiskModel(ABC):
    """Abstract base class for risk models."""

    @property
    @abstractmethod
    def n_centuries(self) -> int:
        """Number of centuries in the simulation."""
        pass

    @property
    @abstractmethod
    def n_branches(self) -> int:
        """Number of branches in the simulation."""
        pass

    @abstractmethod
    def run_simulation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a single simulation of the model.

        Returns:
            tuple containing:
            - results: Array of shape (n_centuries, n_branches) with survival status
            - extinction_centuries: Array of shape (n_branches,) with century of extinction
        """
        pass
