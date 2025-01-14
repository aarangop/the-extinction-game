from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Any
import seaborn as sns
import pandas as pd


class Experiment:
    """Class to handle running experiments with risk models."""

    def __init__(self, model: Any, n_simulations: int = 1000):
        """
        Initialize the experiment.

        Args:
            model: The risk model to use for simulations
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
            results = np.add(results, partial_results)
            # Store the extinction centuries
            self.extinction_centuries[i] = extinction_centuries

        # Calculate the ratio of surviving simulations
        self.results = results / self.n_simulations
        return self.results, self.extinction_centuries

    def get_statistics(self) -> dict:
        """
        Calculate statistics from the experiment results.

        Returns:
            Dictionary containing various statistics:
            - mean_extinction_time: Average time to extinction
            - survival_rate: Percentage of simulations without extinction
            - extinction_percentiles: Various percentiles of extinction times
        """
        if self.extinction_centuries is None:
            raise ValueError("Must run experiment before getting statistics")
        # calculate survival rates as a percentage
        survival_rate = self.results/self.n_simulations

        # Calculate statistics
        mean_extinction_century = np.mean(self.extinction_centuries == -1)

        percentiles = [25, 50, 75, 95]
        extinction_percentiles = np.percentile(
            self.extinction_centuries[self.extinction_centuries != -1],
            percentiles
        )

        return {
            "mean_extinction_time": mean_extinction_century,
            "survival_rate": survival_rate,
            "extinction_percentiles": dict(zip(percentiles, extinction_percentiles))
        }

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

    def plot_results(self) -> plt.Figure:
        """
        Create visualization of multi-risk simulation results.

        Parameters:
        -----------
        results : np.ndarray
            Simulation results array of shape (n_centuries, n_branches)
        extinction_centuries : np.ndarray
            Array of extinction centuries for each simulation run and branch

        Returns:
        --------
        plt.Figure
            Figure containing multiple plots analyzing the results
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Get statistics
        stats_df, survival_by_century = self.get_stats()

        # 1. Average Survival Rate Over Time
        axes[0, 0].fill_between(survival_by_century.century,
                                survival_by_century.q25_survival,
                                survival_by_century.q75_survival,
                                alpha=0.3, color='blue', label='25-75 percentile')
        axes[0, 0].plot(survival_by_century.century,
                        survival_by_century.avg_survival,
                        'b-', label='Mean survival')
        axes[0, 0].set_title('Survival Rate Over Time')
        axes[0, 0].set_xlabel('Century')
        axes[0, 0].set_ylabel('Survival Rate')
        axes[0, 0].legend()

        # 2. Extinction Century Distribution
        extinct_centuries = self.extinction_centuries[self.extinction_centuries != -1]
        if len(extinct_centuries) > 0:
            sns.histplot(data=extinct_centuries, bins=30, ax=axes[0, 1])
            axes[0, 1].axvline(np.median(extinct_centuries), color='r', linestyle='--',
                               label=f'Median: {np.median(extinct_centuries):.1f}')
            axes[0, 1].axvline(np.mean(extinct_centuries), color='g', linestyle='--',
                               label=f'Mean: {np.mean(extinct_centuries):.1f}')
        axes[0, 1].set_title('Distribution of Extinction Centuries')
        axes[0, 1].set_xlabel('Century')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()

        # 3. Survival Heatmap
        sns.heatmap(self.results[:, :min(100, self.results.shape[1])],
                    ax=axes[1, 0],
                    cmap='viridis')
        axes[1, 0].set_title('Survival Heatmap (First 100 Branches)')
        axes[1, 0].set_xlabel('Branch')
        axes[1, 0].set_ylabel('Century')

        # 4. Final State Distribution
        final_state = self.results[-1]
        axes[1, 1].hist(final_state, bins=2, rwidth=0.8)
        axes[1, 1].set_title('Distribution of Final States')
        axes[1, 1].set_xlabel('State (0=Extinct, 1=Survived)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks([0, 1])

        plt.tight_layout()
        return fig

    def estimate_runtime_requirements():
        pass
