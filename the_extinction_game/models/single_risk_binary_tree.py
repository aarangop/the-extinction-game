from typing import Tuple
import numpy as np
import time
from datetime import timedelta


class SingleRiskBinaryTreeModel():

    def __init__(self, n_centuries, alpha):

        self.n_centuries = n_centuries
        self.alpha = alpha
        self.risk_tree = self.build_risk_tree()

    def build_risk_tree(self):
        return build_xrisk_tree(self.n_centuries, initial_risk=0.1, alpha=self.alpha)

    def run_simulation(self):
        return run_survival_simulation(self.risk_tree)

    def run_experiment(self, n_simulations):
        return run_experiment(n_simulations, self.n_centuries, initial_risk=0.1, alpha=self.alpha)

    def estimate_runtime(self, n_simulations):
        return estimate_simulation_runtime(self.n_centuries, n_simulations, initial_risk=0.1, alpha=self.alpha)

    def print_runtime_estimate(self, n_simulations):
        estimate = self.estimate_runtime(n_simulations)
        print("\nSimulation Runtime Estimate")
        print("==========================")
        print(f"Parameters:")
        print(f"  Centuries: {self.n_centuries}")
        print(f"  Simulations: {n_simulations:,}")
        print("\nEstimated Timings:")
        print(f"  Tree Building: {estimate['tree_build_time_seconds']}")
        print(f"  Simulation Runs: {estimate['simulation_time_seconds']}")
        print(f"  Total Runtime: {estimate['estimated_runtime_seconds']}")
        print(f"\nMemory Requirements: {estimate['memory_gb']} GB")
        print("\nBenchmark Details:")
        print(f"  Centuries Tested: {estimate['centuries_tested']}")
        print(f"  Simulations Tested: {estimate['simulations_tested']}")
        print(f"  Benchmark Runtime: {estimate['benchmark_runtime']}")

        if 'warnings' in estimate:
            print("\nWarnings:")
            for warning in estimate['warnings']:
                print(f"  ! {warning}")


def get_xrisk_tree_shape(c_centuries):
    """
    Calculate the shape of an existential risk tree.

    An existential risk tree is a binary tree where each node has two children.
    This function returns a tuple representing the number of levels in the tree
    and the total number of nodes in the tree.

    Parameters:
    n (int): The number of levels in the tree.

    Returns:
    tuple: A tuple containing the number of levels (n) and the total number of nodes (2^(n-1)).
    """
    return c_centuries, 2**(c_centuries-1)


def build_xrisk_tree(n, initial_risk, alpha):
    """
    Builds an existential risk tree matrix.

    Parameters:
    n (int): The depth of the tree. Must be greater than 0.
    initial_risk (float): The initial risk value at the root of the tree. Must be between 0 and 1.
    alpha (float): The risk adjustment factor. Must be between 0 and 1.

    Returns:
    np.ndarray: A 2D numpy array representing the existential risk tree, where each element is a risk value.
    """

    # Validate the inputs
    assert 0 <= alpha <= 1
    assert 0 <= initial_risk <= 1
    assert n > 0

    # Get the dimensions of the matrix
    rows, cols = get_xrisk_tree_shape(n)

    # Initialize the matrix
    tree = np.zeros((rows, cols))
    tree[:] = np.nan

    # Set the initial risk
    tree[0, :] = initial_risk

    # Fill the matrix, starting from the second row, since the first row only has the root element already filled.
    for i in range(1, rows):
        # Compute number of children in current row
        n_children = 2**i
        n_cells_per_child = cols / n_children

        # Iterate through the children in the current row
        for child_col in range(n_children):

            # Since each child can have multiple elements, compute the range of elements that represent this child.
            child_col_range_start = int(child_col*n_cells_per_child)
            child_col_range_end = int(
                child_col_range_start + n_cells_per_child)

            # Get the parent for the current cell, it's always the cell above.
            parent_col = int(child_col * n_cells_per_child)
            parent = tree[i-1, parent_col]

            # Determine the risk direction for this child, should it increase (1) or decrease (-1)?
            direction = 1
            if child_col % 2 == 1:
                direction = -1

            risk = parent + direction * alpha

            # Cap the risk levels
            if risk > 1:
                risk = 1
            if risk < 0:
                risk = 0

            # Set risk level

            # print(f'Cols range: {cols_range_start} to {cols_range_end}')
            tree[i, child_col_range_start:child_col_range_end] = np.float16(
                risk)

    return tree


def set_extinction_event(survival_matrix):
    """
    Determines the extinction event for each branch in the survival matrix and updates the matrix accordingly.

    Parameters:
    survival_matrix (numpy.ndarray): A 2D array where each column represents a branch in the risk tree and each row represents a time step. 
                                     A value of 0 indicates extinction at that time step.

    Returns:
    tuple: A tuple containing:
        - updated_survival_matrix (numpy.ndarray): The updated survival matrix with branches set to zero after the first extinction event.
        - extinction_century (numpy.ndarray): A 1D array where each element represents the century (index) at which the corresponding branch went extinct. 
                                              A value of -1 indicates that the branch survived all the way to the end.
    """

    extinction_century = np.zeros(survival_matrix.shape[1], dtype=int)

    # Check each column and see if any was "lost" (i.e., a civilization went extinct).
    for j in range(survival_matrix.shape[1]):
        # A branch in the tree is represented by each whole column, fetch it.
        branch = survival_matrix[:, j]

        # Find any zeros in the branch
        zeros = np.where(branch == 0)[0]

        # If there are no zeros, then the branch survived all the way to the end.
        if zeros.size == 0:
            extinction_century[j] = -1
            continue

        # Collect the index of the first zero.
        first_zero = zeros[0]

        # Set the extinction century to the place where the first zero was found.
        extinction_century[j] = first_zero

        # Set all entries after the first zero to zero, meaning that after one extinction event, it's game over.
        branch[first_zero:] = 0

        # Replace the branch in the survival matrix.
        survival_matrix[:, j] = branch

    return survival_matrix, extinction_century


def run_survival_simulation(tree) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the survival of civilizations based on an existential risk tree.

    Parameters:
    tree (np.ndarray): A matrix representing the existential risk tree, where each value indicates the risk of extinction for a given period.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - survival_matrix (np.ndarray): A matrix where each value indicates whether the civilization survived (1) or went extinct (0) for each period.
        - extinction_century (np.ndarray): An array indicating the century of extinction for each branch, if applicable.
    """
    # Create a matrix of random numbers
    random_matrix = np.random.random(tree.shape)

    # If the random number in the matrix is higher than the risk in the x-risk tree, then we assume that that civilization survived that period.
    survival_matrix = tree <= random_matrix

    # Convert the boolean matrix to an integer matrix
    survival_matrix = survival_matrix.astype(int)

    # Set the extinction event, by setting all values for each branch after the first 0 to 0, if a 0 is found in the branch.
    survival_matrix, extinction_century = set_extinction_event(
        survival_matrix)

    return survival_matrix, extinction_century


def run_experiment(n_simulations, n_centuries, initial_risk, alpha):
    """
    Run an experiment with the given parameters and return the results.

    Parameters:
    n_simulations (int): The number of simulations to run.
    n_centuries (int): The number of centuries to simulate.
    initial_risk (float): The initial risk value.
    alpha (float): The risk change factor.

    Returns:
    np.ndarray: A 2D numpy array containing the results of the experiment.
    """
    # Build the existential risk tree
    tree = build_xrisk_tree(n_centuries, initial_risk, alpha)

    # Run the simulation
    results = np.zeros((n_simulations, tree.shape[0], tree.shape[1]))
    for i in range(n_simulations):
        partial_results = run_survival_simulation(tree)
        survival_results, _ = set_extinction_event(
            partial_results)
        results[i] = survival_results
    return results


def estimate_simulation_runtime(centuries, n_simulations, initial_risk=0.1, alpha=0.01,
                                benchmark_size=0.01):
    """
    Estimate the runtime of a simulation based on input parameters.

    Parameters:
    -----------
    centuries : int
        Number of centuries to simulate
    n_simulations : int
        Number of simulation runs
    initial_risk : float, optional
        Initial risk value (default 0.1)
    alpha : float, optional
        Risk change factor (default 0.01)
    benchmark_size : float, optional
        Fraction of full simulation to use for benchmarking (default 0.01)

    Returns:
    --------
    dict
        Contains estimated runtime, memory usage, and benchmark details
    """

    # Run a small benchmark first
    # At least 2, at most 4 centuries
    benchmark_centuries = max(2, min(4, centuries))
    benchmark_sims = max(2, int(n_simulations * benchmark_size))

    # Time tree building
    tic = time.time()
    tree = build_xrisk_tree(benchmark_centuries, initial_risk, alpha)
    toc = time.time()
    tree_build_time = toc - tic

    # Time simulation runs
    tic = time.time()
    results = np.zeros((benchmark_sims, tree.shape[0], tree.shape[1]))
    for i in range(benchmark_sims):
        results[i], _ = run_survival_simulation(tree)
    toc = time.time()
    sim_run_time = toc - tic

    # Calculate scaling factors
    # Tree building scales with O(n * 2^(n-1))
    tree_scaling = (centuries * 2**(centuries-1)) / \
        (benchmark_centuries * 2**(benchmark_centuries-1))

    # Simulation runs scale linearly with number of simulations and with tree size
    sim_scaling = (n_simulations / benchmark_sims) * tree_scaling

    # Estimate total runtime
    estimated_tree_time = tree_build_time * tree_scaling
    estimated_sim_time = sim_run_time * sim_scaling
    total_estimated_time = estimated_tree_time + estimated_sim_time

    # Calculate memory requirements
    bytes_per_float = np.float16(0).itemsize
    tree_size = centuries * 2**(centuries-1) * bytes_per_float
    results_size = n_simulations * centuries * \
        2**(centuries-1) * bytes_per_float
    random_matrix_size = centuries * 2**(centuries-1) * bytes_per_float
    total_memory = (tree_size + results_size +
                    random_matrix_size) / (1024**3)  # Convert to GB

    # Format the results
    result = {
        'centuries': centuries,
        'n_simulations': n_simulations,
        'estimated_runtime_seconds': int(total_estimated_time),
        'tree_build_time_seconds': int(estimated_tree_time),
        'simulation_time_seconds': int(estimated_sim_time),
        'memory_gb': total_memory,
        'centuries_tested': benchmark_centuries,
        'simulations_tested': benchmark_sims,
        'benchmark_runtime': tree_build_time+sim_run_time,
    }

    # Add warnings if needed
    if total_memory > 32:  # Arbitrary threshold for warning
        result['warnings'] = [
            "Warning: Memory requirements exceed 32GB. Consider reducing parameters."
        ]
    if total_estimated_time > 24*3600:  # Warning if > 24 hours
        result['warnings'] = result.get('warnings', []) + [
            "Warning: Estimated runtime exceeds 24 hours."
        ]

    return result


def print_runtime_estimate(centuries, n_simulations):
    """
    Print a formatted runtime estimate report.
    """
    estimate = estimate_simulation_runtime(centuries, n_simulations)

    print("\nSimulation Runtime Estimate")
    print("==========================")
    print(f"Parameters:")
    print(f"  Centuries: {centuries}")
    print(f"  Simulations: {n_simulations:,}")
    print("\nEstimated Timings:")
    print(f"  Tree Building: {estimate['tree_build_time']}")
    print(f"  Simulation Runs: {estimate['simulation_time']}")
    print(f"  Total Runtime: {estimate['estimated_runtime']}")
    print(f"\nMemory Requirements: {estimate['memory_gb']} GB")
    print("\nBenchmark Details:")
    for key, value in estimate['benchmark_details'].items():
        print(f"  {key}: {value}")

    if 'warnings' in estimate:
        print("\nWarnings:")
        for warning in estimate['warnings']:
            print(f"  ! {warning}")
