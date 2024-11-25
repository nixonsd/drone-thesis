import pygame
import math
import sympy
import numpy as np
from models.lidar_simulator import simulate_lidar
from models.ekf import ExtendedKalmanFilter

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Player Movement with SLAM")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Clock for controlling the frame rate
clock = pygame.time.Clock()
FPS = 60

# Player settings
player_size = 40
player_x, player_y = 0, 0  # Initial position
player_angle = 0  # Initial angle
player_speed = 5  # Movement speed
rotation_speed = 5  # Rotation speed

# LiDAR settings
lidar_range = 200  # LiDAR detection range
lidar_fov = 90  # Field of view in degrees
obstacles = [(10, 10), (300, 300), (400, 500), (700, 100), (500, 300)]  # Predefined obstacles

# EKF initialization
state_dim = 3  # [x, y, theta]
initial_state = np.zeros((state_dim, 1))  # Initialize all to zero
meas_dim = 3 # [distance, angle] for each obstacle
ekf = ExtendedKalmanFilter(state_dim, meas_dim)
ekf.set_process_noise(np.diag([0.5] * state_dim))  # Process noise for all states
ekf.set_measurement_noise(np.diag([1.0] * meas_dim))  # Measurement noise for all measurements

# # Add obstacle positions to EKF state vector
# for i, (obs_x, obs_y) in enumerate(obstacles):
#   initial_state[3 + 2 * i, 0] = obs_x
#   initial_state[3 + 2 * i + 1, 0] = obs_y

import math

def get_object_position(robot_x, robot_y, robot_angle, r, theta):
  """
  Calculate the global position of an object based on the robot's position, orientation, and lidar readings.

  :param robot_x: Robot's global x-coordinate
  :param robot_y: Robot's global y-coordinate
  :param robot_angle: Robot's global orientation (in radians)
  :param r: Distance to the object from the robot
  :param theta: Angle to the object from the robot (in radians, relative to robot's orientation)
  :return: Tuple (x_global, y_global) representing the global coordinates of the object
  """
  # Incorporate the robot's orientation into the angle
  absolute_angle = theta + robot_angle

  # Calculate the relative Cartesian coordinates
  x_relative = r * math.cos(absolute_angle)
  y_relative = r * math.sin(absolute_angle)

  # Calculate the global coordinates
  x_global = robot_x + x_relative
  y_global = robot_y + y_relative

  return x_global, y_global

def process_lidar_detections(robot_x, robot_y, robot_angle, lidar_detections):
  """
  Process lidar detections and return a flattened list of global positions.

  :param robot_x: Robot's global x-coordinate
  :param robot_y: Robot's global y-coordinate
  :param robot_angle: Robot's global orientation (in radians)
  :param lidar_detections: List of tuples [(r1, theta1), (r2, theta2), ...] in radians
  :return: Flattened list of global coordinates [x1, y1, x2, y2, ...]
  """
  obstacle_coordinates = []
  for detection in lidar_detections:
    r, theta = detection
    # Use the corrected get_object_position function
    x_global, y_global = get_object_position(robot_x, robot_y, robot_angle, r, theta)
    obstacle_coordinates.append(x_global)
    obstacle_coordinates.append(y_global)

  return obstacle_coordinates

def mahalanobis_distance(det_x, det_y, obs_x, obs_y, P):
    """
    Computes the Mahalanobis distance between a detection and an obstacle.
    :param det_x: Detected x-coordinate
    :param det_y: Detected y-coordinate
    :param obs_x: Obstacle x-coordinate in the state
    :param obs_y: Obstacle y-coordinate in the state
    :param P: Covariance matrix (2x2) for the obstacle position
    :return: Mahalanobis distance
    """
    detection = np.array([det_x, det_y]).reshape(-1, 1)
    obstacle = np.array([obs_x, obs_y]).reshape(-1, 1)
    delta = detection - obstacle
    return np.sqrt(delta.T @ np.linalg.inv(P) @ delta)[0, 0]

# def find_correspondence_with_mahalanobis(obstacle_coordinates, state, P, threshold=3.0):
#     """
#     Finds correspondences between detections and obstacles using Mahalanobis distance.
#     :param obstacle_coordinates: List of detected obstacle coordinates [(x1, y1), (x2, y2), ...].
#     :param state: Current state vector of the EKF (includes robot and obstacle states).
#     :param P: Full covariance matrix of the state vector.
#     :param threshold: Mahalanobis distance threshold for valid matches.
#     :return: List of correspondences and unmatched detections.
#     """
#     correspondences = []
#     unmatched_detections = []

#     obstacle_indices = range(3, state.shape[0], 2)  # Indices for obstacles in the state
#     known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

#     for det_x, det_y in obstacle_coordinates:
#         closest_idx = None
#         min_distance = float('inf')

#         for idx, (obs_x, obs_y) in enumerate(known_obstacles):
#             # Extract the covariance matrix for this obstacle (2x2 block)
#             cov_idx = obstacle_indices[idx]
#             P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]

#             # Compute Mahalanobis distance
#             distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

#             if distance < min_distance and distance <= threshold:
#                 min_distance = distance
#                 closest_idx = idx

#         if closest_idx is not None:
#             correspondences.append((closest_idx, (det_x, det_y)))
#         else:
#             unmatched_detections.append((det_x, det_y))

#     return correspondences, unmatched_detections

def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=3.0):
    """
    Finds correspondences and unmatched detections using Mahalanobis distance.
    :param obstacle_coordinates: Flattened array [x1, y1, x2, y2, ...].
    :param state: Current state vector of the EKF (includes robot and obstacle states).
    :param P: Full covariance matrix of the state vector.
    :param threshold: Mahalanobis distance threshold for valid matches.
    :return: List of correspondences [(state_idx, [x, y]), ...] and unmatched detections [x, y, ...].
    """
    # Validate input
    if len(obstacle_coordinates) % 2 != 0:
        raise ValueError(f"Invalid obstacle_coordinates: {obstacle_coordinates}. Must contain an even number of elements.")

    correspondences = []
    unmatched_detections = []

    # Parse obstacle indices in the state
    obstacle_indices = range(3, state.shape[0], 2)
    known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

    # Iterate through flattened obstacle coordinates
    for i in range(0, len(obstacle_coordinates), 2):
        det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
        closest_idx = None
        min_distance = float('inf')

        for idx, (obs_x, obs_y) in enumerate(known_obstacles):
            # Extract covariance matrix for this obstacle
            cov_idx = obstacle_indices[idx]
            P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]

            # Compute Mahalanobis distance
            distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_idx = idx

        if closest_idx is not None:
            correspondences.append((closest_idx, [det_x, det_y]))
        else:
            unmatched_detections.extend([det_x, det_y])  # Flatten unmatched

    return correspondences, unmatched_detections


def motion_model(state, u):
    """
    Motion model for EKF. Supports both symbolic and numerical computations.
    :param state: Current state vector (symbolic or numeric)
    :param u: Control input vector [dx, dy, dtheta]
    :return: Next state vector
    """
    # Extract robot's state
    x, y, theta = state[0, 0], state[1, 0], state[2, 0]
    dx, dy, dtheta = u

    # Update robot's position and orientation
    if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
        # Symbolic computation
        theta_rad = (theta + dtheta) * sympy.pi / 180  # Degrees to radians
        x_next = x + dx * sympy.cos(theta_rad)
        y_next = y + dy * sympy.sin(theta_rad)
        theta_next = theta + dtheta
    else:
        # Numerical computation
        theta_rad = math.radians(theta + dtheta)
        x_next = x + dx * math.cos(theta_rad)
        y_next = y + dy * math.sin(theta_rad)
        theta_next = theta + dtheta

    # Obstacles remain unchanged
    next_state = state.copy()
    next_state[0, 0] = x_next
    next_state[1, 0] = y_next
    next_state[2, 0] = theta_next

    return next_state

# def measurement_model(state):
#     """
#     Measurement model for EKF. Generates measurements for the robot and obstacles.
#     :param state: Current state vector
#     :return: Measurement vector
#     """
#     x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#     measurements = [x, y, theta]  # Robot state measurements

#     # Process obstacle coordinates
#     for i in range(3, state.shape[0], 2):
#         obs_x = state[i, 0]
#         obs_y = state[i + 1, 0]

#         if hasattr(x, 'is_symbol') or hasattr(obs_x, 'is_symbol'):
#             # Symbolic computation
#             distance = sympy.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
#             angle = sympy.atan2(obs_y - y, obs_x - x) - theta
#         else:
#             # Numerical computation
#             distance = math.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
#             angle = math.atan2(obs_y - y, obs_x - x) - theta

#         measurements.extend([distance, angle])

#     return np.array(measurements).reshape((-1, 1))  # Column vector


# def measurement_model(state):
#   """
#   Generate the measurement prediction vector based on the current state.
#   :param state: EKF state vector.
#   :return: Measurement vector z_pred.
#   """
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]  # Robot state
#   measurements = [x, y, theta]  # Start with robot state

#   # Iterate through obstacle positions in the state
#   for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
#       obs_x, obs_y = state[i, 0], state[i + 1, 0]
#       measurements.extend([obs_x, obs_y])

#   return np.array(measurements).reshape((-1, 1))

def measurement_model(state):
    """
    Generate the measurement prediction vector based on the current state.
    :param state: EKF state vector.
    :return: Measurement vector z_pred.
    """
    x, y, theta = state[0, 0], state[1, 0], state[2, 0]  # Robot state
    measurements = [x, y, theta]  # Start with robot state

    # Iterate through obstacle positions in the state
    for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
        obs_x, obs_y = state[i, 0], state[i + 1, 0]
        measurements.extend([obs_x, obs_y])

    return np.array(measurements).reshape((-1, 1))



# def motion_model(state, u):
#   """
#   Motion model for EKF. Supports both symbolic and numerical computations.

#   :param state: Current state vector (symbolic or numeric)
#   :param u: Control input vector [dx, dy, dtheta]
#   :return: Next state vector
#   """
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = u

#   # Handle symbolic computation (if inputs are symbolic)
#   if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
#     theta_rad = (theta + dtheta) * sympy.pi / 180  # Convert to radians for symbolic computation
#     x_next = x + dx * sympy.cos(theta_rad)
#     y_next = y + dy * sympy.sin(theta_rad)
#     theta_next = theta + dtheta
#   else:
#     # Numerical computation
#     theta_rad = np.radians(theta + dtheta)
#     x_next = x + dx * np.cos(theta_rad)
#     y_next = y + dy * np.sin(theta_rad)
#     theta_next = theta + dtheta

#   # Obstacles remain unchanged
#   next_state = state.copy()
#   next_state[0, 0] = x_next
#   next_state[1, 0] = y_next
#   next_state[2, 0] = theta_next
#   return next_state

# def measurement_model(state):
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   measurements = []

#   for i in range(len(obstacles)):
#     obs_x = state[3 + 2 * i, 0]
#     obs_y = state[3 + 2 * i + 1, 0]

#     if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
#       # Symbolic computation
#       distance = sympy.sqrt((obs_x - x)**2 + (obs_y - y)**2)
#       angle = sympy.atan2(obs_y - y, obs_x - x) - theta
#     else:
#       # Numerical computation
#       distance = math.sqrt((obs_x - x)**2 + (obs_y - y)**2)
#       angle = math.atan2(obs_y - y, obs_x - x) - theta

#     measurements.extend([distance, angle])

#   return np.array(measurements).reshape((-1, 1))  # Return as column vector

# Game loop
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
  
  # Key states for movement
  keys = pygame.key.get_pressed()
  if keys[pygame.K_UP]:
    player_x += player_speed * math.cos(player_angle)
    player_y += player_speed * math.sin(player_angle)
  if keys[pygame.K_DOWN]:
    player_x -= player_speed * math.cos(player_angle)
    player_y -= player_speed * math.sin(player_angle)
  if keys[pygame.K_LEFT]:
    player_angle -= math.radians(rotation_speed)  # Convert rotation speed to radians
  if keys[pygame.K_RIGHT]:
    player_angle += math.radians(rotation_speed)  # Convert rotation speed to radians

  # Normalize angle to [-π, π]
  player_angle = (player_angle + math.pi) % (2 * math.pi) - math.pi

  # Initialize z as a copy of the state vector
  z = np.copy(ekf.get_state())  # Ensure z matches the state vector's size

  # Add robot pose to z
  z[0], z[1], z[2] = player_x, player_y, player_angle

  # Simulate LiDAR
  lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)

  # Process LiDAR detections
  obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # Find correspondences and unmatched detections
  correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
    obstacle_coordinates, ekf.get_state(), ekf.get_covariance()
  )

  print("Correspondences:", len(correspondences), correspondences)
  print("Unmatched:", len(unmatched), unmatched)

  for idx, (pos_x, pos_y) in correspondences:
    z[idx + 3], z[idx + 1 + 3] = pos_x, pos_y
  
  # Add unmatched detections to EKF state
  for u in unmatched:  # Add only if there are unmatched detections
    ekf.add_dimension(initial_value=u, process_noise=0.001, measurement_noise=0.01)
    z = np.append(z, u)

  # Convert z to numpy array
  z = np.array(z).reshape((-1, 1))
  
  # Debug print statements
  print("EKF state (x):", len(ekf.get_state()))
  print("Measurement vector (z):", len(z), z)

  # Initial EKF prediction
  ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))
  
  # Update EKF with the measurement vector
  ekf.update(z, measurement_model)
  
  # # Simulate LiDAR
  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)

  # # Process LiDAR detections
  # obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # # Find correspondences and unmatched detections
  # correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
  #     obstacle_coordinates, ekf.get_state(), ekf.get_covariance()
  # )

  # print(correspondences, unmatched)

  # # Add unmatched detections to EKF state
  # for u in unmatched:  # Add only if there are unmatched detections
  #   ekf.add_dimension(initial_value=u, process_noise=0.001, measurement_noise=0.01)

  # # Initial EKF prediction
  # ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))

  # # Construct the measurement vector z
  # z = [player_x, player_y, player_angle]  # Start with robot state

  # # Add placeholders for all obstacles in the state
  # for idx in range(3, ekf.get_state().shape[0], 2):  # Obstacles start at index 3
  #   z.extend([np.nan, np.nan])

  # # Update z with matched obstacles
  # for state_idx, (det_x, det_y) in correspondences:
  #     # Find the position of the obstacle in z
  #     obstacle_idx = 3 + (state_idx - 3)  # Offset in z
  #     z[obstacle_idx] = det_x
  #     z[obstacle_idx + 1] = det_y

  # print(len(unmatched) // 2)

  # # Add placeholders for unmatched detections (if any new obstacles were added)
  # for i in range(len(unmatched) // 2):
  #     z.extend([unmatched[i * 2], unmatched[i * 2 + 1]])

  # # Convert z to numpy array
  # z = np.array(z).reshape((-1, 1))
  # print('x', ekf.x)
  # print('z', z)

  # # Update EKF with the measurement vector
  # ekf.update(z, measurement_model)


  # # Simulate LiDAR
  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
  
  # # Process LiDAR detections
  # obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # print(obstacle_coordinates)

  # # Initial EKF prediction
  # ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))

  # # Find correspondences and unmatched detections
  # correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
  #     obstacle_coordinates, ekf.get_state(), ekf.get_covariance()
  # )

  # # Add new obstacles to the EKF state
  # if unmatched:  # Add only if there are unmatched detections
  #     ekf.add_dimension(initial_value=unmatched, process_noise=0.001, measurement_noise=0.01)

  # # Construct the measurement vector z
  # z = [player_x, player_y, player_angle]  # Start with robot state

  # # Add matched detections to z
  # for _, (det_x, det_y) in correspondences:
  #     z.extend([det_x, det_y])

  # # Add unmatched detections directly
  # z.extend(unmatched)

  # # Convert z to numpy array
  # z = np.array(z).reshape((-1, 1))

  # # Update EKF with the measurement vector
  # ekf.update(z, measurement_model)


  
  # # Process LiDAR detections
  # obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # # Predict step
  # ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))

  # # Find correspondences and unmatched detections
  # correspondences, unmatched = find_correspondence_with_mahalanobis(
  #     obstacle_coordinates, ekf.get_state(), ekf.get_covariance()
  # )

  # # Add new obstacles to the EKF state
  # for det_x, det_y in unmatched:
  #     ekf.add_dimension(initial_value=[det_x, det_y], process_noise=0.001, measurement_noise=0.01)

  # # Construct the measurement vector z
  # # Initialize z with robot state
  # z = [player_x, player_y, player_angle]

  # # Get indices of obstacles in the state vector
  # obstacle_indices = range(3, ekf.get_state().shape[0], 2)

  # # Prepare measurements for obstacles
  # z_obstacles = [np.nan] * len(obstacle_indices)

  # # Add measurements for matched obstacles
  # for obs_idx, (det_x, det_y) in correspondences:
  #     z_obstacles[obs_idx] = (det_x, det_y)

  # # Add measurements to z
  # for i, idx in enumerate(obstacle_indices):
  #     if z_obstacles[i] is None:
  #         z.extend([np.nan, np.nan])  # Placeholder for undetected obstacles
  #     else:
  #         z.extend(z_obstacles[i])

  # # Convert to numpy array
  # z = np.array(z).reshape((-1, 1))

  # # Update EKF
  # ekf.update(z, measurement_model)

  
  # # Simulate LiDAR
  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
  # # Process LiDAR detections
  # obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # # Initial state and prediction
  # ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))
  
  # # Find correspondences and unmatched detections
  # correspondences, unmatched = find_correspondence_with_mahalanobis(obstacle_coordinates, ekf.get_state(), ekf.get_covariance())

  # # Add new obstacles to the EKF state
  # for det_x, det_y in unmatched:
  #     ekf.add_dimension(initial_value=[det_x, det_y], process_noise=0.001, measurement_noise=0.01)

  # # Build the measurement vector z
  # z = np.array([player_x, player_y, player_angle])  # Start with robot state
  # for obs_idx, (det_x, det_y) in correspondences:
  #     z = np.append(z, [det_x, det_y])

  # # Add unmatched detections to z as new obstacles
  # for det_x, det_y in unmatched:
  #     z = np.append(z, [det_x, det_y])

  # # Update EKF with measurements
  # ekf.update(z.reshape((-1, 1)), measurement_model)

  # # Simulate LiDAR
  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
  # # Process LiDAR detections
  # obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)
  
  # # Initial state and prediction
  # ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))

  # for coord in obstacle_coordinates:
  #   ekf.add_dimension(initial_value=coord, process_noise=0.001, measurement_noise=0.01)

  # # Update with measurements
  # z = np.array([player_x, player_y, player_angle])
  # z = np.append(z, obstacle_coordinates)
  # ekf.update(z.reshape((-1, 1)), measurement_model)

  
  # Construct z as a 2D column vector
  # z = ([player_x, player_y, player_angle] + ).reshape((-1, 1))

  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
  # if lidar_detections:
  #   # Flatten lidar detections into a single column vector
  #   z = np.array([val for detection in lidar_detections for val in detection]).reshape((-1, 1))
  #   ekf.update(z, measurement_model)


  # SLAM: Prediction and Update
  # state = np.array([player_x, player_y, player_angle] + [coord for obstacle in obstacles for coord in obstacle]).reshape((-1, 1))
  # ekf.predict(lambda s, u: motion_model(state, [0, 0, 0]))

  # Simulate LiDAR
  # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)

  # # Construct z as a 2D column vector
  # z = np.array([[distance, angle] for distance, angle in lidar_detections]).flatten().reshape((-1, 1))

  # # Update EKF
  # ekf.update(z, measurement_model)

  # Clear screen
  screen.fill(WHITE)

  # Draw player as a square
  half_size = player_size // 2
  corners = [
    (
      player_x + half_size * math.cos(player_angle) - half_size * math.sin(player_angle),
      player_y + half_size * math.sin(player_angle) + half_size * math.cos(player_angle),
    ),
    (
      player_x - half_size * math.cos(player_angle) - half_size * math.sin(player_angle),
      player_y - half_size * math.sin(player_angle) + half_size * math.cos(player_angle),
    ),
    (
      player_x - half_size * math.cos(player_angle) + half_size * math.sin(player_angle),
      player_y - half_size * math.sin(player_angle) - half_size * math.cos(player_angle),
    ),
    (
      player_x + half_size * math.cos(player_angle) + half_size * math.sin(player_angle),
      player_y + half_size * math.sin(player_angle) - half_size * math.cos(player_angle),
    ),
  ]
  pygame.draw.polygon(screen, BLUE, corners)

  # Draw obstacles
  for obstacle in obstacles:
    pygame.draw.circle(screen, RED, obstacle, 10)

  # Draw LiDAR detections
  for detection in lidar_detections:
    distance, angle = detection
    # Use angle directly as it is already in radians
    end_x = player_x + distance * math.cos(player_angle + angle)
    end_y = player_y + distance * math.sin(player_angle + angle)
    pygame.draw.line(screen, GREEN, (player_x, player_y), (end_x, end_y), 2)

  # Display SLAM state
  font = pygame.font.Font(None, 36)
  slam_text = font.render(f"SLAM State: x={player_x:.2f}, y={player_y:.2f}, angle={math.degrees(player_angle):.2f}", True, BLACK)
  screen.blit(slam_text, (10, 10))

  # Update display
  pygame.display.flip()
  clock.tick(FPS)

pygame.quit()


# import pygame
# import math
# import sympy
# import numpy as np
# from models.lidar_simulator import simulate_lidar
# from models.ekf import ExtendedKalmanFilter

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Player Movement with SLAM")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)

# # Clock for controlling the frame rate
# clock = pygame.time.Clock()
# FPS = 60

# # Player settings
# player_size = 40
# player_x, player_y = 0, 0  # Initial position
# player_angle = 0  # Initial angle
# player_speed = 5  # Movement speed
# rotation_speed = 5  # Rotation speed

# # LiDAR settings
# lidar_range = 200  # LiDAR detection range
# lidar_fov = 90  # Field of view in degrees
# obstacles = [(300, 300), (400, 500), (700, 100), (500, 300)]  # Predefined obstacles

# # EKF initialization
# state_dim = 3 + 2 * len(obstacles)  # [x, y, theta] + [x_1, y_1, ...]
# initial_state = np.zeros((state_dim, 1))  # Initialize all to zero
# meas_dim = 2 * len(obstacles)  # [distance, angle] for each obstacle
# ekf = ExtendedKalmanFilter(state_dim, meas_dim)
# ekf.set_process_noise(np.diag([0.5] * state_dim))  # Process noise for all states
# ekf.set_measurement_noise(np.diag([1.0] * meas_dim))  # Measurement noise for all measurements


# meas_dim = 2 * len(obstacles)  # [distance, angle] for each obstacle
# ekf = ExtendedKalmanFilter(state_dim, meas_dim)
# ekf.set_process_noise(np.diag([0.5] * state_dim))  # Process noise for all states
# ekf.set_measurement_noise(np.diag([1.0] * meas_dim))  # Measurement noise for all measurements


# # Add obstacle positions to EKF state vector
# for i, (obs_x, obs_y) in enumerate(obstacles):
#   initial_state[3 + 2 * i, 0] = obs_x
#   initial_state[3 + 2 * i + 1, 0] = obs_y

# import math

# def get_object_position(robot_x, robot_y, robot_angle, r, theta):
#   """
#   Calculate the global position of an object based on the robot's position, orientation, and lidar readings.

#   :param robot_x: Robot's global x-coordinate
#   :param robot_y: Robot's global y-coordinate
#   :param robot_angle: Robot's global orientation (in radians)
#   :param r: Distance to the object from the robot
#   :param theta: Angle to the object from the robot (in radians, relative to robot's orientation)
#   :return: Tuple (x_global, y_global) representing the global coordinates of the object
#   """
#   # Incorporate the robot's orientation into the angle
#   absolute_angle = theta + robot_angle

#   # Calculate the relative Cartesian coordinates
#   x_relative = r * math.cos(absolute_angle)
#   y_relative = r * math.sin(absolute_angle)

#   # Calculate the global coordinates
#   x_global = robot_x + x_relative
#   y_global = robot_y + y_relative

#   return x_global, y_global

# def process_lidar_detections(robot_x, robot_y, robot_angle, lidar_detections):
#   """
#   Process lidar detections and return a flattened list of global positions.

#   :param robot_x: Robot's global x-coordinate
#   :param robot_y: Robot's global y-coordinate
#   :param robot_angle: Robot's global orientation (in radians)
#   :param lidar_detections: List of tuples [(r1, theta1), (r2, theta2), ...] in radians
#   :return: Flattened list of global coordinates [x1, y1, x2, y2, ...]
#   """
#   obstacle_coordinates = []
#   for detection in lidar_detections:
#     r, theta = detection
#     # Use the corrected get_object_position function
#     x_global, y_global = get_object_position(robot_x, robot_y, robot_angle, r, theta)
#     obstacle_coordinates.extend([x_global, y_global])
#   return obstacle_coordinates

# def find_closest_obstacles(detected_coordinates, known_obstacles, threshold=50.0):
#   """
#   Associate LiDAR detections with known obstacles.

#   :param detected_coordinates: List of detected obstacle coordinates [x1, y1, x2, y2, ...].
#   :param known_obstacles: List of known obstacles [[x1, y1], [x2, y2], ...].
#   :param threshold: Maximum distance to consider a match (default = 50.0).
#   :return: List of associated obstacles [(det_x, det_y, matched_obstacle_index), ...].
#   """
#   associated_obstacles = []
#   for i in range(0, len(detected_coordinates), 2):
#     det_x, det_y = detected_coordinates[i], detected_coordinates[i + 1]
#     closest_idx = None
#     min_distance = float('inf')

#     for idx, (obs_x, obs_y) in enumerate(known_obstacles):
#       distance = np.sqrt((det_x - obs_x) ** 2 + (det_y - obs_y) ** 2)
#       if distance < min_distance and distance <= threshold:
#         min_distance = distance
#         closest_idx = idx

#     if closest_idx is not None:
#       associated_obstacles.append((det_x, det_y, closest_idx))
#   return associated_obstacles

# # def motion_model(state, u):
# #   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
# #   dx, dy, dtheta = u
# #   theta_rad = np.radians(theta + dtheta)
# #   x_next = x + dx * np.cos(theta_rad)
# #   y_next = y + dy * np.sin(theta_rad)
# #   theta_next = theta + dtheta
# #   # Obstacles remain unchanged
# #   next_state = state.copy()
# #   next_state[0, 0] = x_next
# #   next_state[1, 0] = y_next
# #   next_state[2, 0] = theta_next
# #   return next_state

# def motion_model(state, u):
#   """
#   Motion model for EKF. Supports both symbolic and numerical computations.

#   :param state: Current state vector (symbolic or numeric)
#   :param u: Control input vector [dx, dy, dtheta]
#   :return: Next state vector
#   """
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = u

#   # Handle symbolic computation (if inputs are symbolic)
#   if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
#     theta_rad = (theta + dtheta) * sympy.pi / 180  # Convert to radians for symbolic computation
#     x_next = x + dx * sympy.cos(theta_rad)
#     y_next = y + dy * sympy.sin(theta_rad)
#     theta_next = theta + dtheta
#   else:
#     # Numerical computation
#     theta_rad = np.radians(theta + dtheta)
#     x_next = x + dx * np.cos(theta_rad)
#     y_next = y + dy * np.sin(theta_rad)
#     theta_next = theta + dtheta

#   # Obstacles remain unchanged
#   next_state = state.copy()
#   next_state[0, 0] = x_next
#   next_state[1, 0] = y_next
#   next_state[2, 0] = theta_next
#   return next_state



# # def measurement_model(state):
# #   # Assume the LiDAR detects distances and angles relative to the player's position
# #   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
# #   measurements = []
# #   for i in range(len(obstacles)):
# #     obs_x = state[3 + 2 * i, 0]
# #     obs_y = state[3 + 2 * i + 1, 0]
# #     distance = math.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
# #     angle = math.degrees(math.atan2(obs_y - y, obs_x - x)) - theta
# #     measurements.append([distance, angle])
# #   return np.array(measurements).flatten()

# def measurement_model(state):
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   measurements = []

#   for i in range(len(obstacles)):
#     obs_x = state[3 + 2 * i, 0]
#     obs_y = state[3 + 2 * i + 1, 0]

#     if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
#       # Symbolic computation
#       distance = sympy.sqrt((obs_x - x)**2 + (obs_y - y)**2)
#       angle = sympy.atan2(obs_y - y, obs_x - x) - theta
#     else:
#       # Numerical computation
#       distance = math.sqrt((obs_x - x)**2 + (obs_y - y)**2)
#       angle = math.atan2(obs_y - y, obs_x - x) - theta

#     measurements.extend([distance, angle])

#   return np.array(measurements).reshape((-1, 1))  # Return as column vector

# # Game loop
# running = True
# while running:
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
  
#   # Key states for movement
#   keys = pygame.key.get_pressed()
#   if keys[pygame.K_UP]:
#     player_x += player_speed * math.cos(player_angle)
#     player_y += player_speed * math.sin(player_angle)
#   if keys[pygame.K_DOWN]:
#     player_x -= player_speed * math.cos(player_angle)
#     player_y -= player_speed * math.sin(player_angle)
#   if keys[pygame.K_LEFT]:
#     player_angle -= math.radians(rotation_speed)  # Convert rotation speed to radians
#   if keys[pygame.K_RIGHT]:
#     player_angle += math.radians(rotation_speed)  # Convert rotation speed to radians

#   # Normalize angle to [-π, π]
#   player_angle = (player_angle + math.pi) % (2 * math.pi) - math.pi

#   # # SLAM: Prediction and Update
#   # state = np.array([player_x, player_y, player_angle] + [coord for obstacle in obstacles for coord in obstacle]).reshape((-1, 1))
#   # ekf.predict(lambda s, u: motion_model(state, [0, 0, 0]))

#   # # Simulate LiDAR
#   # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)  
#   # obstacle_coordinates = process_lidar_detections(player_x, player_y, lidar_detections)
  
#   # print(obstacle_coordinates)
  
#   # SLAM: Prediction
#   ekf.predict(lambda s, u: motion_model(s, [0, 0, 0]))

#   # Simulate LiDAR
#   lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
#   # Process LiDAR detections
#   obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

#   print(obstacle_coordinates)
  
#   # # SLAM: Prediction and Update
#   # state = np.array([player_x, player_y, player_angle] + [coord for obstacle in obstacles for coord in obstacle]).reshape((-1, 1))
#   # ekf.predict(lambda s, u: motion_model(state, [0, 0, 0]))

#   # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)
#   # if lidar_detections:
#   #   # Flatten lidar detections into a single column vector
#   #   z = np.array([val for detection in lidar_detections for val in detection]).reshape((-1, 1))
#   #   ekf.update(z, measurement_model)


#   # SLAM: Prediction and Update
#   # state = np.array([player_x, player_y, player_angle] + [coord for obstacle in obstacles for coord in obstacle]).reshape((-1, 1))
#   # ekf.predict(lambda s, u: motion_model(state, [0, 0, 0]))

#   # Simulate LiDAR
#   # lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)

#   # # Construct z as a 2D column vector
#   # z = np.array([[distance, angle] for distance, angle in lidar_detections]).flatten().reshape((-1, 1))

#   # # Update EKF
#   # ekf.update(z, measurement_model)

#   # Clear screen
#   screen.fill(WHITE)

#   # Draw player as a square
#   half_size = player_size // 2
#   corners = [
#     (
#       player_x + half_size * math.cos(player_angle) - half_size * math.sin(player_angle),
#       player_y + half_size * math.sin(player_angle) + half_size * math.cos(player_angle),
#     ),
#     (
#       player_x - half_size * math.cos(player_angle) - half_size * math.sin(player_angle),
#       player_y - half_size * math.sin(player_angle) + half_size * math.cos(player_angle),
#     ),
#     (
#       player_x - half_size * math.cos(player_angle) + half_size * math.sin(player_angle),
#       player_y - half_size * math.sin(player_angle) - half_size * math.cos(player_angle),
#     ),
#     (
#       player_x + half_size * math.cos(player_angle) + half_size * math.sin(player_angle),
#       player_y + half_size * math.sin(player_angle) - half_size * math.cos(player_angle),
#     ),
#   ]
#   pygame.draw.polygon(screen, BLUE, corners)

#   # Draw obstacles
#   for obstacle in obstacles:
#     pygame.draw.circle(screen, RED, obstacle, 10)

#   # Draw LiDAR detections
#   for detection in lidar_detections:
#     distance, angle = detection
#     # Use angle directly as it is already in radians
#     end_x = player_x + distance * math.cos(player_angle + angle)
#     end_y = player_y + distance * math.sin(player_angle + angle)
#     pygame.draw.line(screen, GREEN, (player_x, player_y), (end_x, end_y), 2)

#   # Display SLAM state
#   font = pygame.font.Font(None, 36)
#   slam_text = font.render(f"SLAM State: x={player_x:.2f}, y={player_y:.2f}, angle={math.degrees(player_angle):.2f}", True, BLACK)
#   screen.blit(slam_text, (10, 10))

#   # Update display
#   pygame.display.flip()
#   clock.tick(FPS)

# pygame.quit()


# import pygame
# import numpy as np
# from models.slam import SLAM

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("SLAM Game")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)

# # Clock for controlling the frame rate
# clock = pygame.time.Clock()
# FPS = 60

# # Player settings
# player_size = 40
# player_x, player_y = WIDTH // 2, HEIGHT // 2  # Initial position
# player_angle = 0  # Initial angle
# player_speed = 5  # Movement speed
# rotation_speed = 5  # Rotation speed

# # LiDAR settings
# lidar_range = 200  # LiDAR detection range
# lidar_fov = 90  # Field of view in degrees
# obstacles = [(300, 300), (400, 500), (700, 100), (500, 300)]  # Predefined obstacles

# # Initialize SLAM
# player_pos = [player_x, player_y, np.radians(player_angle)]
# process_noise = np.diag([0.1] * (3 + 2 * len(obstacles)))
# measurement_noise = np.diag([0.5, 0.1] * len(obstacles))
# slam = SLAM(player_pos, obstacles, process_noise, measurement_noise)

# # Game loop
# running = True
# while running:
#   # Event handling
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False

#   # Key states
#   keys = pygame.key.get_pressed()

#   # Control inputs [dx, dy, dtheta]
#   dx = player_speed * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
#   dtheta = np.radians(rotation_speed * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]))
#   control = [dx, 0, dtheta]

#   # Perform SLAM prediction step
#   slam.predict(control)

  # Simulate LiDAR sensor
  # lidar_measurements = []
  # for obstacle in obstacles:
  #   dx = obstacle[0] - player_x
  #   dy = obstacle[1] - player_y
  #   distance = np.sqrt(dx**2 + dy**2)
  #   angle = np.degrees(np.arctan2(dy, dx)) - player_angle
  #   angle = (angle + 180) % 360 - 180  # Normalize angle to [-180, 180]
  #   if distance <= lidar_range and abs(angle) <= lidar_fov / 2:
  #     lidar_measurements.append((distance, angle))

  # Perform SLAM update step
  # if lidar_measurements:
  #   slam.update(lidar_measurements)

  # # Get updated SLAM state
  # state = slam.get_state()
  # player_estimated_x, player_estimated_y, player_estimated_theta = state[0, 0], state[1, 0], state[2, 0]
  # estimated_obstacles = [
  #   (state[3 + 2 * i, 0], state[3 + 2 * i + 1, 0]) for i in range(len(obstacles))
  # ]

#   # Clear screen
#   screen.fill(WHITE)

#   # Draw player
#   half_size = player_size // 2
#   angle_rad = np.radians(player_angle)
#   corners = [
#     (
#       player_x + half_size * np.cos(angle_rad) - half_size * np.sin(angle_rad),
#       player_y + half_size * np.sin(angle_rad) + half_size * np.cos(angle_rad),
#     ),
#     (
#       player_x - half_size * np.cos(angle_rad) - half_size * np.sin(angle_rad),
#       player_y - half_size * np.sin(angle_rad) + half_size * np.cos(angle_rad),
#     ),
#     (
#       player_x - half_size * np.cos(angle_rad) + half_size * np.sin(angle_rad),
#       player_y - half_size * np.sin(angle_rad) - half_size * np.cos(angle_rad),
#     ),
#     (
#       player_x + half_size * np.cos(angle_rad) + half_size * np.sin(angle_rad),
#       player_y + half_size * np.sin(angle_rad) - half_size * np.cos(angle_rad),
#     ),
#   ]
#   pygame.draw.polygon(screen, BLUE, corners)

#   # Draw real obstacles
#   for obstacle in obstacles:
#     pygame.draw.circle(screen, RED, obstacle, 10)

#   # Draw estimated obstacles
#   # for est_obstacle in estimated_obstacles:
#   #   pygame.draw.circle(screen, GREEN, (int(est_obstacle[0]), int(est_obstacle[1])), 5)

#   # Display SLAM state and measurements
#   font = pygame.font.Font(None, 36)
#   # slam_text = font.render(f"SLAM Player Pos: [{player_estimated_x:.2f}, {player_estimated_y:.2f}, {np.degrees(player_estimated_theta):.2f}°]", True, BLACK)
#   # screen.blit(slam_text, (10, 10))

#   lidar_text = font.render(f"LiDAR Detections: {len(lidar_measurements)}", True, BLACK)
#   screen.blit(lidar_text, (10, 50))

#   # Update display
#   pygame.display.flip()

#   # Cap the frame rate
#   clock.tick(FPS)

# pygame.quit()



# import pygame
# import math
# from models.lidar_simulator import simulate_lidar

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Player Square Movement with LiDAR Sensor")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)

# # Clock for controlling the frame rate
# clock = pygame.time.Clock()
# FPS = 60

# # Player settings
# player_size = 40
# player_x, player_y = WIDTH // 2, HEIGHT // 2  # Initial position
# player_angle = 0  # Initial angle
# player_speed = 5  # Movement speed
# rotation_speed = 5  # Rotation speed

# # LiDAR settings
# lidar_range = 200  # LiDAR detection range
# lidar_fov = 90  # Field of view in degrees
# obstacles = [(300, 300), (400, 500), (700, 100), (500, 300)]  # Predefined obstacles

# # Game loop
# running = True

# while running:
#   # Event handling
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False

#   # Key states
#   keys = pygame.key.get_pressed()

#   # Control array [x, y, angle]
#   control_array = [player_x, player_y, player_angle]

#   # Movement and rotation logic
#   if keys[pygame.K_UP]:
#     player_x += player_speed * math.cos(math.radians(player_angle))
#     player_y += player_speed * math.sin(math.radians(player_angle))
#   if keys[pygame.K_DOWN]:
#     player_x -= player_speed * math.cos(math.radians(player_angle))
#     player_y -= player_speed * math.sin(math.radians(player_angle))
#   if keys[pygame.K_LEFT]:
#     player_angle -= rotation_speed
#   if keys[pygame.K_RIGHT]:
#     player_angle += rotation_speed

#   # Normalize angle to keep it between 0 and 360
#   player_angle %= 360
#   control_array = [round(player_x, 2), round(player_y, 2), round(player_angle, 2)]

#   # Clear screen
#   screen.fill(WHITE)

#   # Draw player
#   half_size = player_size // 2
#   angle_rad = math.radians(player_angle)
#   corners = [
#     (
#       player_x + half_size * math.cos(angle_rad) - half_size * math.sin(angle_rad),
#       player_y + half_size * math.sin(angle_rad) + half_size * math.cos(angle_rad),
#     ),
#     (
#       player_x - half_size * math.cos(angle_rad) - half_size * math.sin(angle_rad),
#       player_y - half_size * math.sin(angle_rad) + half_size * math.cos(angle_rad),
#     ),
#     (
#       player_x - half_size * math.cos(angle_rad) + half_size * math.sin(angle_rad),
#       player_y - half_size * math.sin(angle_rad) - half_size * math.cos(angle_rad),
#     ),
#     (
#       player_x + half_size * math.cos(angle_rad) + half_size * math.sin(angle_rad),
#       player_y + half_size * math.sin(angle_rad) - half_size * math.cos(angle_rad),
#     ),
#   ]
#   pygame.draw.polygon(screen, BLUE, corners)

#   # Draw obstacles
#   for obstacle in obstacles:
#     pygame.draw.circle(screen, RED, obstacle, 10)

#   # Simulate LiDAR
#   lidar_detections = simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov)

#   # Draw LiDAR detections
#   for detection in lidar_detections:
#     distance, angle = detection
#     end_x = player_x + distance * math.cos(math.radians(player_angle + angle))
#     end_y = player_y + distance * math.sin(math.radians(player_angle + angle))
#     pygame.draw.line(screen, GREEN, (player_x, player_y), (end_x, end_y), 2)

#   # Display the control array and lidar data
#   font = pygame.font.Font(None, 36)
#   control_text = font.render(f"Control Array: {control_array}", True, BLACK)
#   screen.blit(control_text, (10, 10))

#   lidar_text = font.render(f"LiDAR: {len(lidar_detections)} detections", True, BLACK)
#   screen.blit(lidar_text, (10, 50))

#   # Update the display
#   pygame.display.flip()

#   # Cap the frame rate
#   clock.tick(FPS)

# pygame.quit()

# import numpy as np
# from models.ekf import ExtendedKalmanFilter

# # Define time step
# dt = 1.0  # 1 second

# sigma = 0.5

# initial_state = np.array([0, 0])  # assume x
# control_inputs = np.array([[1, 0], [1, 0], [1, 0]])  # assume speed

# def add_noise(data, sigma):
#     """
#     Add Gaussian noise to the data to simulate sensor noise.
#     Handles both scalar and iterable data.
#     """
#     if np.isscalar(data):  # Check if the input is a scalar (float or int)
#         noise = np.random.normal(0, sigma)
#         return data + noise
#     else:  # Handle iterable data (e.g., list or numpy array)
#         noise = np.random.normal(0, sigma, size=len(data))
#         return data + noise

# def f(state, control_input):
#     return np.array([state[0] + dt * control_input[0], 0])

# def h(state):
#     return state

# # Initialize the EKF
# ekf = ExtendedKalmanFilter(state_dim=2, meas_dim=2, initial_state=initial_state)

# # Define process and measurement noise covariances
# ekf.set_process_noise(np.eye(2) * 0.01)  # Small process noise
# ekf.set_measurement_noise(np.eye(2) * sigma)  # Moderate measurement noise

# # Create the array of measurements
# measurements = np.array([[add_noise(i, sigma), 0] for i in range(1, 4)])

# # Run EKF for three time steps
# for k, (z, u) in enumerate(zip(measurements, control_inputs)):
#     print(f"\nTime Step {k+1}:")
    
#     # Predict step
#     ekf.predict(f, u=u)
#     print("Predicted State:")
#     print(ekf.get_state())
#     print("Predicted Covariance:")
#     print(ekf.get_covariance())

#     # Update step with measurement
#     ekf.update(z, h)  # Pass `z` directly; it's already a 1D array
#     print("Updated State:")
#     print(ekf.get_state())
#     print("Updated Covariance:")
#     print(ekf.get_covariance())

# import numpy as np
# from models.ekf import ExtendedKalmanFilter

# # Define time step
# dt = 1.0  # 1 second

# sigma = 0.5

# initial_state = np.array([0, 0]) # assume x
# control_inputs = np.array([[1, 0], [1, 0], [1, 0]]) # assume speed

# def add_noise(data, sigma):
#     """
#     Add Gaussian noise to the data to simulate sensor noise.
#     Handles both scalar and iterable data.
#     """
#     if np.isscalar(data):  # Check if the input is a scalar (float or int)
#         noise = np.random.normal(0, sigma)
#         return data + noise
#     else:  # Handle iterable data (e.g., list or numpy array)
#         noise = np.random.normal(0, sigma, size=len(data))
#         return data + noise

# def f(state, control_input):
#     return np.array([state[0] + dt * control_input[0], 0])

# def h(state):
#     return state

# # Initialize the EKF
# ekf = ExtendedKalmanFilter(state_dim=2, meas_dim=2, initial_state=initial_state)

# # Define process and measurement noise covariances
# ekf.set_process_noise(np.eye(2) * 0.001)  # Small process noise
# ekf.set_measurement_noise(np.eye(2) * sigma)             # Moderate measurement noise

# # # Initialize the covariance matrix
# # ekf.P = np.array([[1, 0], [0, 1]])  # Initial uncertainty

# # measurements = [add_noise(x, sigma) for x in np.arange(1, 4, 1)]

# # Create the array of measurements
# measurements = np.array([[add_noise(i, sigma), 0] for i in range(1, 3)])

# # Run EKF for three time steps
# for k, (z, u) in enumerate(zip(measurements, control_inputs)):
#     print(f"\nTime Step {k+1}:")
    
#     # Predict step
#     ekf.predict(f, u=u)
#     print("Predicted State:")
#     print(ekf.get_state())
#     print("Predicted Covariance:")
#     print(ekf.get_covariance())

#     # Update step with measurement
#     z = np.array([z, 0])  # Convert measurement to column vector
#     ekf.update(z, h)
#     print("Updated State:")
#     print(ekf.get_state())
#     print("Updated Covariance:")
#     print(ekf.get_covariance())

# # State transition function
# def f(state, control_input):
#     print('state', state)
#     """
#     State transition function.
#     x_next = [x + v * dt, v]
#     """
#     x, v = state[0, 0], state[1, 0]
#     v_control = control_input[0, 0] if control_input is not None else v
#     return np.array([[x + v_control * dt], [v_control]])

# # Measurement function
# def h(state):
#     """
#     Measurement function.
#     Measures position (x) only.
#     """
#     x, _ = state
#     return np.array([[x]])

# # Initialize the EKF
# ekf = ExtendedKalmanFilter(state_dim=2, meas_dim=1, initial_state=np.array([[0], [0]]))  # Starting at x=0, v=0

# # Define process and measurement noise covariances
# ekf.set_process_noise(np.array([[0.01, 0], [0, 0.01]]))  # Small process noise
# ekf.set_measurement_noise(np.array([[0.1]]))             # Moderate measurement noise

# # Initialize the covariance matrix
# ekf.P = np.array([[1, 0], [0, 1]])  # Initial uncertainty

# # Simulated measurements (position only, with some noise)
# measurements = [1.1, 2.2, 3.1]
# control_inputs = [np.array([[1]]), np.array([[1]]), np.array([[1]])]  # Constant velocity control input

# # Run EKF for three time steps
# for k, (z, u) in enumerate(zip(measurements, control_inputs)):
#     print(f"\nTime Step {k+1}:")
    
#     # Predict step
#     ekf.predict(f, u=u)
#     print("Predicted State:")
#     print(ekf.get_state())
#     print("Predicted Covariance:")
#     print(ekf.get_covariance())
    
#     # Update step with measurement
#     z = np.array([[z]])  # Convert measurement to column vector
#     ekf.update(z, h)
#     print("Updated State:")
#     print(ekf.get_state())
#     print("Updated Covariance:")
#     print(ekf.get_covariance())