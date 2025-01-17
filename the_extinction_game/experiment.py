from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Any
import seaborn as sns
import pandas as pd
from .models.risk_model_interface import RiskModel


class Experiment:
    """Class to handle running experiments with risk models."""

    def __init__(self, model: RiskModel, n_simulations: int = 1000):
        """
        Initialize the experiment.

        Args:
            model: The risk model to use for simulations (must implement RiskModel interface)
            n_simulations: Number of simulations to run (default: 1000)
        """
        self.model = model
        self.n_simulations = n_simulations
        self.results = None
        self.extinction_centuries = None

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the experiment for the specified number of simulations.

        Returns:
            Tuple containing:
            - Array of aggregated results
            - Array of extinction centuries for each simulation
        """
        results = np.zeros(
            (self.model.n_centuries, self.model.n_branches))
        self.extinction_centuries = np.ndarray(
            (self.n_simulations, self.model.n_branches))

        # Run simulations
        for i in range(self.n_simulations):
            # Run individual simulation
            partial_results, extinction_centuries = self.model.run_simulation()
            # Add up the results
            results += partial_results
            # Store the extinction centuries
            self.extinction_centuries[i] = extinction_centuries

        # Calculate the ratio of surviving simulations
        self.results = results / self.n_simulations
        return self.results, self.extinction_centuries

    def get_stats(self):
        """
        Generate comprehensive statistics from multi-risk simulation results.

        Parameters:
        -----------
        results : np.ndarray
            Simulation results array of shape (n_centuries, n_branches)
        extinction_centuries : np.ndarray
            Array of extinction centuries for each simulation run and branch

        Returns:
        --------
        pd.DataFrame
            DataFrame containing various statistics about the simulation results
        """

        # Calculate survival statistics per century
        survival_by_century = pd.DataFrame({
            'century': range(1, self.results.shape[0] + 1),
            'avg_survival': self.results.mean(axis=1),
            'q25_survival': np.percentile(self.results, 25, axis=1),
            'q75_survival': np.percentile(self.results, 75, axis=1)
        })

        # Process extinction centuries
        # Filter out -1 values (surviving branches) for extinction statistics
        extinct_mask = self.extinction_centuries != -1
        extinct_centuries = self.extinction_centuries[extinct_mask]

        stats = {
            'overall_survival_rate': (self.extinction_centuries == -1).mean(),
            'total_simulations': len(self.extinction_centuries.flatten()),
            'total_extinctions': len(extinct_centuries),
            'mean_extinction_century': np.mean(extinct_centuries) if len(extinct_centuries) > 0 else np.nan,
            'median_extinction_century': np.median(extinct_centuries) if len(extinct_centuries) > 0 else np.nan,
            'std_extinction_century': np.std(extinct_centuries) if len(extinct_centuries) > 0 else np.nan,
            'earliest_extinction': np.min(extinct_centuries) if len(extinct_centuries) > 0 else np.nan,
            'latest_extinction': np.max(extinct_centuries) if len(extinct_centuries) > 0 else np.nan,
            'q25_extinction_century': np.percentile(extinct_centuries, 25) if len(extinct_centuries) > 0 else np.nan,
            'q75_extinction_century': np.percentile(extinct_centuries, 75) if len(extinct_centuries) > 0 else np.nan
        }

        # Convert stats to DataFrame
        stats_df = pd.DataFrame([stats])

        return stats_df, survival_by_century

    def plot_survival_rate(self, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the survival rate over time.
        This method creates a figure with a single subplot that shows the average survival rate over time,
        along with the 25th to 75th percentile range.
        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes objects of the plot.
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)

        # Get statistics
        stats_df, survival_by_century = self.get_stats()

        # 1. Average Survival Rate Over Time (covering two columns)
        # Make the first ax span two columns
        ax.fill_between(survival_by_century.century,
                        survival_by_century.q25_survival,
                        survival_by_century.q75_survival,
                        alpha=0.3, color='blue', label='25-75 percentile')
        ax.plot(survival_by_century.century,
                survival_by_century.avg_survival,
                'b-', label='Mean survival')
        ax.set_title('Survival Rate Over Time')
        ax.set_xlabel('Century')
        ax.set_ylabel('Survival Rate')
        ax.legend()

        if fig is not None:
            return fig, ax

        return ax

    def plot_extinction_century_histogram(self, ax: plt.Axes = None) -> plt.Figure:
        """
        Figure containing a histogram of extinction centuries with median and mean lines.
        """

        # Get data
        _, extinct_centuries = self.get_stats()
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)

        extinct_centuries = self.extinction_centuries[self.extinction_centuries != -1]
        if len(extinct_centuries) > 0:
            sns.histplot(data=extinct_centuries,
                         bins=30, ax=ax, stat='count')
            ax.axvline(np.median(extinct_centuries), color='r', linestyle='--',
                       label=f'Median: {np.median(extinct_centuries):.1f}')
            ax.axvline(np.mean(extinct_centuries), color='g', linestyle='--',
                       label=f'Mean: {np.mean(extinct_centuries):.1f}')
        ax.set_title('Distribution of Extinction Centuries')
        ax.set_xlabel('Century')
        ax.set_ylabel('Count')
        ax.legend()

        plt.tight_layout()

        if fig is not None:
            return fig, ax

        return ax

    def estimate_runtime_requirements():
        pass
