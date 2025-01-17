import random
import numpy as np
from the_extinction_game.models.risk_model_interface import RiskModel
from the_extinction_game.models.single_risk_binary_tree import run_survival_simulation


class DisjointEventsConstantRiskModel(RiskModel):
    """A model for simulating the survival of an entity over a number of centuries with a constant risk of extinction.

    Attributes:
      n_centuries (int): The number of centuries to simulate.
      n_branches (int): The number of branches in the model (default is 1).
      survival (numpy.ndarray): An array to store survival outcomes for each century.
      initial_risk (float): The initial risk of extinction per century.
      centuries (numpy.ndarray): An array representing each century in the simulation.
      century_risk (numpy.ndarray): An array to store the computed risk for each century.

    Methods:
      n_centuries: Returns the number of centuries.
      n_branches: Returns the number of branches.
      compute_survival_probability(century: float) -> np.ndarray: Computes the survival probability for a given century.
      run_simulation() -> tuple: Runs the extinction simulation and returns survival outcomes and the extinction century.
    """

    def __init__(self, n_centuries: int, initial_risk: float):
        self._n_centuries = n_centuries
        self._n_branches = 1
        self.survival = np.ndarray([self._n_centuries, self.n_branches])
        self.initial_risk = initial_risk
        self.centuries = np.arange(1, self._n_centuries + 1)
        self.survival_probability_per_century = self.compute_survival_probability(
            self.centuries)

    @property
    def n_centuries(self):
        return self._n_centuries

    @property
    def n_branches(self):
        return self._n_branches

    def compute_survival_probability(self, century: float) -> np.ndarray:
        """
        Compute the survival probability over a given number of centuries.

        Parameters:
        century (float): The number of centuries over which to compute the survival probability.

        Returns:
        np.ndarray: An array containing the survival probability after the given number of centuries.
        """
        return (1 - self.initial_risk)

    def run_simulation(self):
        """
        Runs the extinction simulation over a number of centuries.

        This method simulates the survival of an entity over a specified number of centuries.
        It computes the survival probability for each century, generates random numbers to
        determine survival outcomes, and identifies the century in which extinction occurs.

        Returns:
          tuple: A tuple containing:
            - survival (numpy.ndarray): An array indicating survival (1) or extinction (0) for each century.
            - extinction_century (int): The century in which extinction occurs, or -1 if survival continues.
        """
        # Generate random numbers for each century
        dice_throw = np.random.random(self.n_centuries)

        # Mark "survived" centuries
        survival = (
            dice_throw < self.survival_probability_per_century).astype(int)

        # Find extinction events, by default assume no extinction
        extinction_century = -1

        # The appearance of the first zero means extinction, find it
        zeros = np.where(survival == 0)[0]

        # If zeros is an array with size > 0, then at least one extinction event occurred, fetch it
        if zeros.size > 0:
            extinction_century = zeros[0] + 1

        # If there was an extinction event, set the survival status to zero for all subsequent centuries
        if extinction_century != -1:
            survival[extinction_century:] = 0

        # Reshape survival to match the shape (n_centuries, n_branches)
        return survival.reshape(self._n_centuries, self._n_branches), extinction_century
