import numpy as np

def gradient_descent_algorithm(users, NUM_USERS, NUM_ANTENNAS, COVERAGE_RADIUS, GRID_SIZE):
    """
    Places antennas using gradient descent optimization.

    Parameters:
    - users: List of user positions (list of tuples).
    - NUM_USERS: Number of users.
    - NUM_ANTENNAS: Number of antennas to place.
    - COVERAGE_RADIUS: Coverage radius of each antenna.
    - GRID_SIZE: Size of the grid area.

    Returns:
    - antennas: List of antennas' positions.
    - cost_history: List of cost function values over iterations.
    """
    # Algorithm-specific parameters
    LEARNING_RATE = 0.01
    ITERATIONS = 1000
    LAMBDA = 0.1  # Regularization parameter for interference

    # Convert users to numpy array for vectorized operations
    users_array = np.array(users)  # Shape: (NUM_USERS, 2)

    # Initialize antenna positions randomly
    antennas = np.random.uniform(0, GRID_SIZE, (NUM_ANTENNAS, 2))  # Shape: (NUM_ANTENNAS, 2)

    cost_history = []

    for iteration in range(ITERATIONS):
        # Compute distances between antennas and users
        # Shape: (NUM_ANTENNAS, NUM_USERS)
        distances = np.linalg.norm(antennas[:, np.newaxis, :] - users_array[np.newaxis, :, :], axis=2)

        # Signal strengths (1 / (1 + distance))
        signals = np.where(distances <= COVERAGE_RADIUS, 1 / (1 + distances), 0)

        # Coverage: Users covered by at least one antenna
        max_signals = np.max(signals, axis=0)
        covered_users = max_signals > 0
        coverage_score = np.sum(covered_users) / NUM_USERS

        # Total interference: Sum of signals where multiple antennas cover the same user
        interference = np.sum(np.sum(signals, axis=0) - max_signals)

        # Cost function
        cost = -(coverage_score - LAMBDA * interference)
        cost_history.append(cost)

        # Compute gradients
        gradients = np.zeros_like(antennas)  # Shape: (NUM_ANTENNAS, 2)

        for i in range(NUM_ANTENNAS):
            for j in range(NUM_USERS):
                if distances[i, j] <= COVERAGE_RADIUS:
                    # Partial derivative of the cost with respect to antenna position
                    diff = antennas[i] - users_array[j]
                    dist = distances[i, j] + 1e-6  # Add small value to prevent division by zero
                    signal_grad = -1 / (1 + dist) ** 2 * (diff / dist)
                    # Adjust for coverage and interference
                    num_covering_antennas = np.sum(distances[:, j] <= COVERAGE_RADIUS)
                    if num_covering_antennas == 1:
                        # Only this antenna covers the user
                        gradients[i] += signal_grad / NUM_USERS
                    else:
                        # Multiple antennas cover the user (interference)
                        gradients[i] += (signal_grad / NUM_USERS) + (LAMBDA * signal_grad)

        # Update antenna positions
        antennas -= LEARNING_RATE * gradients

        # Ensure antennas stay within grid boundaries
        antennas = np.clip(antennas, 0, GRID_SIZE)

        # Optional: Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Cost = {cost:.4f}")

    # Convert antennas back to list of tuples
    antennas_list = [tuple(coord) for coord in antennas]

    return antennas_list, cost_history
