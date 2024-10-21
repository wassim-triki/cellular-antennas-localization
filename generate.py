import json
import random

# Set random seed for reproducibility
random.seed(42)

# Parameters to be shared across all algorithms
shared_parameters = {
    "GRID_SIZE": 500,       # Size of the area (100x100 units)
    "NUM_USERS": 100,        # Number of users to be randomly distributed
    "NUM_ANTENNAS": 20,      # Number of antennas to place
    "COVERAGE_RADIUS": 20   # Coverage radius of each antenna
}

# Generate random user positions within the grid
def generate_user_positions(num_users, grid_size):
    users = []
    for _ in range(num_users):
        x = round(random.uniform(0, grid_size),2)
        y = round(random.uniform(0, grid_size),2)
        users.append({"x": x, "y": y})
    return users

# Main function to create the config.json file
def create_config_file(filename='config.json'):
    # Generate user positions
    users = generate_user_positions(
        num_users=shared_parameters["NUM_USERS"],
        grid_size=shared_parameters["GRID_SIZE"]
    )
    
    # Structure the data to be saved
    data = {
        "parameters": shared_parameters,
        "users": users
    }
    
    # Save the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Configuration file '{filename}' has been generated successfully.")

# Run the script to generate the config.json file
if __name__ == '__main__':
    create_config_file()
