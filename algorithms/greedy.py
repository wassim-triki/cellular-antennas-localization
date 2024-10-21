import math

def greedy_algorithm(users, NUM_USERS, NUM_ANTENNAS, COVERAGE_RADIUS):
    """
    Places antennas using a greedy algorithm.
    
    Parameters:
    - users: List of user positions (list of tuples).
    - NUM_USERS: Number of users.
    - NUM_ANTENNAS: Number of antennas to place.
    - COVERAGE_RADIUS: Coverage radius of each antenna.
    
    Returns:
    - antennas: List of antennas' positions.
    - covered_users: List indicating which users are covered.
    """
    antennas = []
    covered_users = [False] * NUM_USERS  # Track which users are covered

    while len(antennas) < NUM_ANTENNAS:
        best_location = -1
        max_coverage = 0

        # Iterate through users to find the best location for the next antenna
        for i in range(NUM_USERS):
            if not covered_users[i]:  # If this user is not covered
                coverage_count = sum(
                    1 for j in range(NUM_USERS)
                    if not covered_users[j] and distance(users[i], users[j]) <= COVERAGE_RADIUS
                )

                # Update best location if more coverage is found
                if coverage_count > max_coverage:
                    max_coverage = coverage_count
                    best_location = i

        # Place the antenna at the best location found
        if best_location != -1:
            antennas.append(users[best_location])  # Store antenna location
            # Mark users covered by this antenna
            for j in range(NUM_USERS):
                if not covered_users[j] and distance(users[best_location], users[j]) <= COVERAGE_RADIUS:
                    covered_users[j] = True  # Mark this user as covered
        else:
            # No more uncovered users; break the loop
            break

    return antennas, covered_users

def distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    # Use integer indices since points are tuples
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    # Or use math.hypot for simplicity
    # return math.hypot(point2[0] - point1[0], point2[1] - point1[1])
