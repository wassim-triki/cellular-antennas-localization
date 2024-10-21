# benchmark.py

import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from algorithms.genetic import genetic_algorithm
from algorithms.greedy import greedy_algorithm
from algorithms.gradient_descent import gradient_descent_algorithm

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
np.random.seed(42)  # For numpy random numbers

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
    greedy_antennas, greedy_covered_users = greedy_algorithm(
        NUM_ANTENNAS=NUM_ANTENNAS,
        COVERAGE_RADIUS=COVERAGE_RADIUS,
        NUM_USERS=NUM_USERS,
        users=users
    )
    greedy_time = time.time() - start_time
    print(f"Greedy Algorithm completed in {greedy_time:.2f} seconds.")

    # Run Gradient Descent Algorithm
    print("\nRunning Gradient Descent Algorithm...")
    start_time = time.time()
    gd_antennas, gd_cost_history = gradient_descent_algorithm(
        users=users,
        NUM_USERS=NUM_USERS,
        NUM_ANTENNAS=NUM_ANTENNAS,
        COVERAGE_RADIUS=COVERAGE_RADIUS,
        GRID_SIZE=GRID_SIZE
    )
    gd_time = time.time() - start_time
    print(f"Gradient Descent Algorithm completed in {gd_time:.2f} seconds.")

    # Plot the results side by side
    plot_results_three_algorithms(
        ga_solution, greedy_antennas, gd_antennas, users,
        GRID_SIZE, COVERAGE_RADIUS
    )

def plot_results_three_algorithms(
    ga_solution, greedy_antennas, gd_antennas,
    users, GRID_SIZE, COVERAGE_RADIUS
):
    """
    Plots the antenna positions and user locations for all three algorithms
    side by side, and adds a table comparing their complexities and solution
    types.
    """
    import matplotlib.gridspec as gridspec

    # Compute the metrics for the table
    # Temporal Complexity
    ga_temporal_complexity = (
        "O(GENERATIONS × POP_SIZE × USERS × ANTENNAS)"
    )
    greedy_temporal_complexity = "O(ANTENNAS × USERS²)"
    gd_temporal_complexity = "O(ITERATIONS × ANTENNAS × USERS)"

    # Spatial Complexity
    ga_spatial_complexity = "O(POP_SIZE × ANTENNAS)"
    greedy_spatial_complexity = "O(USERS + ANTENNAS)"
    gd_spatial_complexity = "O(ANTENNAS + USERS)"

    # Type of Solution
    ga_solution_type = "Multiple (Stochastic)"
    greedy_solution_type = "Unique (Deterministic)"
    gd_solution_type = "Unique (Deterministic)"

    # Create figure with gridspec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[5, 0.1, 1])

    # Subplots for each algorithm
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[0, 2])

    # Unpack user coordinates
    user_xs, user_ys = zip(*users)

    # Plot for Genetic Algorithm
    ax0.scatter(user_xs, user_ys, c='blue', label='Users', alpha=0.5)
    ga_antenna_xs, ga_antenna_ys = zip(*ga_solution)
    ax0.scatter(ga_antenna_xs, ga_antenna_ys, c='red', label='Antennas')
    for antenna in ga_solution:
        circle = plt.Circle(
            antenna, COVERAGE_RADIUS, color='red',
            fill=False, alpha=0.3
        )
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
    ax1.scatter(
        greedy_antenna_xs, greedy_antenna_ys,
        c='green', label='Antennas'
    )
    for antenna in greedy_antennas:
        circle = plt.Circle(
            antenna, COVERAGE_RADIUS, color='green',
            fill=False, alpha=0.3
        )
        ax1.add_artist(circle)
    ax1.set_xlim(0, GRID_SIZE)
    ax1.set_ylim(0, GRID_SIZE)
    ax1.set_title('Greedy Algorithm')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()

    # Plot for Gradient Descent Algorithm
    ax2.scatter(user_xs, user_ys, c='blue', label='Users', alpha=0.5)
    gd_antenna_xs, gd_antenna_ys = zip(*gd_antennas)
    ax2.scatter(
        gd_antenna_xs, gd_antenna_ys,
        c='purple', label='Antennas'
    )
    for antenna in gd_antennas:
        circle = plt.Circle(
            antenna, COVERAGE_RADIUS, color='purple',
            fill=False, alpha=0.3
        )
        ax2.add_artist(circle)
    ax2.set_xlim(0, GRID_SIZE)
    ax2.set_ylim(0, GRID_SIZE)
    ax2.set_title('Gradient Descent Algorithm')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.legend()

    # Remove unused axes
    ax_empty = plt.subplot(gs[1, :])
    ax_empty.axis('off')  # Hide this axis

    # Create the table
    ax_table = plt.subplot(gs[2, :])
    ax_table.axis('off')  # Hide the axis

    # Table data
    cell_text = [
        [
            ga_temporal_complexity,
            greedy_temporal_complexity,
            gd_temporal_complexity
        ],
        [
            ga_spatial_complexity,
            greedy_spatial_complexity,
            gd_spatial_complexity
        ],
        [
            ga_solution_type,
            greedy_solution_type,
            gd_solution_type
        ]
    ]
    rows = [
        'Temporal Complexity',
        'Spatial Complexity',
        'Type of Solution'
    ]
    columns = [
        'Genetic Algorithm',
        'Greedy Algorithm',
        'Gradient Descent'
    ]

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
