import json
import random
import math
import matplotlib.pyplot as plt
import time
from algorithms.genetic import genetic_algorithm
from algorithms.greedy import greedy_algorithm

# Load shared data from config.json
with open('config.json', 'r') as f:
    data = json.load(f)

# Extract shared parameters
GRID_SIZE = data['parameters']['GRID_SIZE']
NUM_USERS = data['parameters']['NUM_USERS']
NUM_ANTENNAS = data['parameters']['NUM_ANTENNAS']
COVERAGE_RADIUS = data['parameters']['COVERAGE_RADIUS']

# Load user positions
users = [(user['x'], user['y']) for user in data['users']]

# Set random seed for reproducibility
random.seed(42)

def main():
    # Run Genetic Algorithm
    print("Running Genetic Algorithm...")
    start_time = time.time()
    ga_solution, ga_fitness_history = genetic_algorithm(
        GRID_SIZE=GRID_SIZE,
        NUM_USERS=NUM_USERS,
        NUM_ANTENNAS=NUM_ANTENNAS,
        COVERAGE_RADIUS=COVERAGE_RADIUS,
        users=users
    )
    ga_time = time.time() - start_time
    print(f"Genetic Algorithm completed in {ga_time:.2f} seconds.")

    # Run Greedy Algorithm
    print("\nRunning Greedy Algorithm...")
    start_time = time.time()
    greedy_antennas, greedy_covered_users  = greedy_algorithm(
        NUM_ANTENNAS=NUM_ANTENNAS,
        COVERAGE_RADIUS=COVERAGE_RADIUS,
        NUM_USERS=NUM_USERS,
        users=users
    )
    greedy_time = time.time() - start_time
    print(f"Greedy Algorithm completed in {greedy_time:.2f} seconds.")

    # Plot the results side by side
    plot_results_side_by_side(ga_solution, greedy_antennas, users, GRID_SIZE, COVERAGE_RADIUS)

def plot_results_side_by_side(ga_solution, greedy_antennas, users, GRID_SIZE, COVERAGE_RADIUS):
    """
    Plots the antenna positions and user locations for both algorithms side by side,
    and adds a table comparing their complexities and solution types.
    """
    import matplotlib.gridspec as gridspec

    # Compute the metrics for the table
    # Temporal Complexity
    ga_temporal_complexity = f"O(GENERATIONS × POP_SIZE × USERS × ANTENNAS)"
    greedy_temporal_complexity = f"O(ANTENNAS × USERS²)"

    # Spatial Complexity
    ga_spatial_complexity = f"O(POP_SIZE × ANTENNAS)"
    greedy_spatial_complexity = f"O(USERS + ANTENNAS)"

    # Type of Solution
    ga_solution_type = "Multiple (Stochastic)"
    greedy_solution_type = "Unique (Deterministic)"

    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[5, 0.1, 1])

    # First subplot for Genetic Algorithm
    ax0 = plt.subplot(gs[0, 0])
    # Second subplot for Greedy Algorithm
    ax1 = plt.subplot(gs[0, 1])

    # Unpack user coordinates
    user_xs, user_ys = zip(*users)

    # Plot for Genetic Algorithm
    ax0.scatter(user_xs, user_ys, c='blue', label='Users', alpha=0.5)
    ga_antenna_xs, ga_antenna_ys = zip(*ga_solution)
    ax0.scatter(ga_antenna_xs, ga_antenna_ys, c='red', label='Antennas')
    for antenna in ga_solution:
        circle = plt.Circle(antenna, COVERAGE_RADIUS, color='red', fill=False, alpha=0.3)
        ax0.add_artist(circle)
    ax0.set_xlim(0, GRID_SIZE)
    ax0.set_ylim(0, GRID_SIZE)
    ax0.set_title('Genetic Algorithm')
    ax0.set_xlabel('X Coordinate')
    ax0.set_ylabel('Y Coordinate')
    ax0.legend()

    # Plot for Greedy Algorithm
    ax1.scatter(user_xs, user_ys, c='blue', label='Users', alpha=0.5)
    greedy_antenna_xs, greedy_antenna_ys = zip(*greedy_antennas)
    ax1.scatter(greedy_antenna_xs, greedy_antenna_ys, c='red', label='Antennas')
    for antenna in greedy_antennas:
        circle = plt.Circle(antenna, COVERAGE_RADIUS, color='red', fill=False, alpha=0.3)
        ax1.add_artist(circle)
    ax1.set_xlim(0, GRID_SIZE)
    ax1.set_ylim(0, GRID_SIZE)
    ax1.set_title('Greedy Algorithm')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()

    # Remove unused axes (we have 3 rows but only need 2 plots)
    ax_empty = plt.subplot(gs[1, :])
    ax_empty.axis('off')  # Hide this axis

    # Create the table
    ax_table = plt.subplot(gs[2, :])
    ax_table.axis('off')  # Hide the axis

    # Table data
    cell_text = [
        [ga_temporal_complexity, greedy_temporal_complexity],
        [ga_spatial_complexity, greedy_spatial_complexity],
        [ga_solution_type, greedy_solution_type]
    ]
    rows = ['Temporal Complexity', 'Spatial Complexity', 'Type of Solution']
    columns = ['Genetic Algorithm', 'Greedy Algorithm']

    # Create the table
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust the scaling as needed

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
