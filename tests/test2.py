import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import heapq

# Environment settings
ENV_SIZE = 100  # Size of the environment (100x100 units)
OBSTACLE_COUNT = 20  # Number of obstacles
OBSTACLE_SIZE = 5    # Radius of obstacles
GRID_RESOLUTION = 1  # Grid cell size

# Drone settings
DRONE_SPEED = 1.0    # Units per time step
TIME_STEPS = 500     # Max number of time steps for the simulation

# Sensor noise characteristics
GPS_NOISE_STD = 2.0  # Standard deviation of GPS noise

# Kalman Filter settings
KF_PROCESS_NOISE = 1e-2
KF_MEASUREMENT_NOISE = 1e-1

# Generate obstacles
def generate_obstacles(env_size, obstacle_count, obstacle_size):
    obstacles = []
    for _ in range(obstacle_count):
        x = np.random.uniform(obstacle_size, env_size - obstacle_size)
        y = np.random.uniform(obstacle_size, env_size - obstacle_size)
        obstacles.append({'position': np.array([x, y]), 'size': obstacle_size})
    return obstacles

# Check for collision
def check_collision(position, obstacles):
    for obs in obstacles:
        distance = np.linalg.norm(position - obs['position'])
        if distance < obs['size']:
            return True
    return False

# Simulate sensor data
def simulate_sensors(true_position):
    # Simulate GPS data with noise
    gps_position = true_position + np.random.normal(0, GPS_NOISE_STD, 2)
    return gps_position

# Kalman Filter setup
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0  # Time step
    # State transition matrix
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    # Covariance matrices
    kf.P *= 10.0
    kf.R = np.eye(2) * KF_MEASUREMENT_NOISE
    kf.Q = np.eye(4) * KF_PROCESS_NOISE
    # Initial state
    kf.x = np.array([0, 0, 0, 0])
    return kf

# A* pathfinding algorithm
def astar(start, goal, obstacles, grid_size, grid_resolution):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Create grid
    grid_width = int(grid_size / grid_resolution)
    grid_height = int(grid_size / grid_resolution)
    grid = np.zeros((grid_width, grid_height))

    # Mark obstacles on the grid
    for obs in obstacles:
        x_min = int((obs['position'][0] - obs['size']) / grid_resolution)
        x_max = int((obs['position'][0] + obs['size']) / grid_resolution)
        y_min = int((obs['position'][1] - obs['size']) / grid_resolution)
        y_max = int((obs['position'][1] + obs['size']) / grid_resolution)
        x_min = max(0, x_min)
        x_max = min(grid_width - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(grid_height - 1, y_max)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if np.linalg.norm((np.array([x, y]) * grid_resolution) - obs['position']) <= obs['size']:
                    grid[x, y] = 1  # Mark as obstacle

    # A* algorithm
    start_node = (int(start[0] / grid_resolution), int(start[1] / grid_resolution))
    goal_node = (int(goal[0] / grid_resolution), int(goal[1] / grid_resolution))

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, start_node, None))
    came_from = {}
    cost_so_far = {start_node: 0}

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)

        if current == goal_node:
            # Reconstruct path
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from[parent]
            path.reverse()
            return [ (node[0] * grid_resolution, node[1] * grid_resolution) for node in path ]

        came_from[current] = parent

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height:
                    if grid[neighbor[0], neighbor[1]] == 0:
                        neighbors.append(neighbor)

        for neighbor in neighbors:
            new_cost = cost_so_far[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (priority, new_cost, neighbor, current))

    print("No path found!")
    return None

# Simulate drone navigation
def simulate_drone_navigation():
    # Initialize positions
    true_positions = []
    estimated_positions = []
    gps_positions = []
    last_true_position = np.array([0.0, 0.0])

    # Generate obstacles
    obstacles = generate_obstacles(ENV_SIZE, OBSTACLE_COUNT, OBSTACLE_SIZE)

    # Create Kalman Filter
    kf = create_kalman_filter()

    # Target position
    target_position = np.array([ENV_SIZE - 1, ENV_SIZE - 1])

    # Compute path using A*
    path = astar(last_true_position, target_position, obstacles, ENV_SIZE, GRID_RESOLUTION)
    if path is None:
        print("Failed to find a path to the target.")
        return

    # Convert path to waypoints
    waypoints = path.copy()

    # Simulation loop
    for t in range(TIME_STEPS):
        # Check if waypoints are exhausted
        if not waypoints:
            print("Reached target at time step:", t)
            break

        # Get next waypoint
        next_waypoint = np.array(waypoints[0])

        # Calculate direction towards next waypoint
        direction = next_waypoint - last_true_position
        distance_to_waypoint = np.linalg.norm(direction)

        if distance_to_waypoint < DRONE_SPEED:
            # Waypoint reached
            last_true_position = next_waypoint
            waypoints.pop(0)
            continue
        else:
            # Move towards waypoint
            direction = direction / distance_to_waypoint * DRONE_SPEED
            true_position = last_true_position + direction

        # Simulate sensors
        gps_position = simulate_sensors(true_position)

        # Update Kalman Filter
        kf.predict()
        z = gps_position  # Measurement
        kf.update(z)
        estimated_position = kf.x[:2]

        # Store positions
        true_positions.append(true_position.copy())
        estimated_positions.append(estimated_position.copy())
        gps_positions.append(gps_position.copy())

        # Update last position
        last_true_position = true_position.copy()

    else:
        print("Reached maximum time steps without reaching the target.")

    # Visualization
    visualize_navigation(true_positions, estimated_positions, gps_positions, obstacles, target_position, path)

# Visualization function
def visualize_navigation(true_positions, estimated_positions, gps_positions, obstacles, target_position, path):
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)
    gps_positions = np.array(gps_positions)

    plt.figure(figsize=(10, 10))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'k-', label='True Path')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b-', label='Estimated Path')
    plt.scatter(gps_positions[:, 0], gps_positions[:, 1], c='r', s=10, label='GPS Measurements')

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle(obs['position'], obs['size'], color='orange', fill=True, alpha=0.5)
        plt.gca().add_artist(circle)

    # Plot target position
    plt.scatter(target_position[0], target_position[1], c='g', marker='X', s=100, label='Target Position')

    # Plot planned path
    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'c--', label='Planned Path')

    plt.xlim(0, ENV_SIZE)
    plt.ylim(0, ENV_SIZE)
    plt.legend()
    plt.title('Drone Navigation with A* Path Planning')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()

# Run the simulation
simulate_drone_navigation()
