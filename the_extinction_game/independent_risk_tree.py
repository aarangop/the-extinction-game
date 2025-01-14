import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple

from the_extinction_game.basic_risk_tree import build_xrisk_tree, run_survival_simulation, set_extinction_event


class MultiRiskModel:

    def __init__(self,
                 risk_categories: Dict[str, Tuple[float, float]],
                 n_centuries: int):
        """
        Initialize multi-risk model.

        Parameters:
        risk_categories: Dict mapping category names to (initial_risk, alpha) tuples
        n_centuries: Number of centuries to simulate
        """
        self.risk_categories = risk_categories
        self.n_categories = len(risk_categories)
        self.n_centuries = n_centuries
        self.n_branches = 2**(n_centuries-1)

        # Build risk trees
        self.risk_trees = self.build_risk_trees()

    def build_risk_trees(self) -> Dict[str, np.ndarray]:
        """Build risk trees for all categories."""

        self.risk_trees = np.zeros(
            (self.n_categories, self.n_centuries, self.n_branches))

        for i, (_, (initial_risk, alpha)) in enumerate(self.risk_categories.items()):
            self.risk_trees[i] = build_xrisk_tree(
                self.n_centuries, initial_risk, alpha
            )
        return self.risk_trees

    def consolidate_results(self, category_results: np.ndarray) -> np.ndarray:
        """
        Consolidate results across categories.

        Parameters:
        category_results: Array of shape (n_categories, n_centuries, n_branches)
                          containing survival status (1=survived, 0=extinct)

        Returns:
        np.ndarray: Array of shape (n_centuries, n_branches)
                   containing consolidated survival status
        """
        consolidated_results = np.prod(category_results, axis=0)

        consolidated_results, extinction_century = set_extinction_event(
            consolidated_results)

        return consolidated_results, extinction_century

    def run_simulation(self) -> np.ndarray:
        """
        Runs a multi-risk simulation across different risk categories and consolidates the results.

        This method iterates through the risk trees for each category, runs a survival simulation for each tree,
        and stores the partial results and extinction centuries. It then consolidates the results across all categories.

        Returns:
          tuple: A tuple containing:
            - consolidated_results (np.ndarray): The consolidated results of the simulation.
            - consolidated_extinction_centuries (np.ndarray): The extinction centuries for the consolidated results.
            - category_results (np.ndarray): The partial results for each category.
            - extinction_centuries (np.ndarray): The extinction centuries for each category.
        """
        category_results = np.zeros(self.risk_trees.shape)
        consolidated_results = np.zeros(self.risk_trees.shape[1:])
        extinction_centuries = np.zeros((self.n_categories, self.n_branches))

        # Iterate through the risk trees for each category
        for i, tree in enumerate(self.risk_trees):
            # Run a simulation for the category.
            # The run_survival_simulation function will return the extinction results for a single category, along with the century at which every branch went extinct (or not (-1))
            partial_results, extinction_century = run_survival_simulation(tree)
            # Store partial results
            category_results[i] = partial_results
            extinction_centuries[i] = extinction_century

        # Consolidate results, and obtain an extinction century for the consolidated results
        consolidated_results, consolidated_extinction_centuries = self.consolidate_results(
            category_results)

        return (consolidated_results, consolidated_extinction_centuries)
