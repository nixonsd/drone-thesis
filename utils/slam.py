import numpy as np
import math
import sympy
from utils.ekf import find_correspondence_with_mahalanobis_flat
from models.ekf import ExtendedKalmanFilter
from utils.lidar import process_lidar_detections

def motion_model(state, u):
  """
  Motion model for EKF.
  :param state: Current state vector [x, y, theta, ...].
  :param u: Control input vector [dx, dy, dtheta].
  :return: Updated state vector.
  """
  x, y, theta = state[0, 0], state[1, 0], state[2, 0]
  dx, dy, dtheta = u

  if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
    # Symbolic computation
    theta_rad = (theta + dtheta) * sympy.pi / 180
    x_next = x + dx * sympy.cos(theta_rad)
    y_next = y + dy * sympy.sin(theta_rad)
    theta_next = theta + dtheta
  else:
    # Numerical computation
    theta_rad = math.radians(theta + dtheta)
    x_next = x + dx * math.cos(theta_rad)
    y_next = y + dy * math.sin(theta_rad)
    theta_next = theta + dtheta

  next_state = state.copy()
  next_state[0, 0], next_state[1, 0], next_state[2, 0] = x_next, y_next, theta_next

  return next_state

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

def predict(ekf, motion_model):
  """
  Perform the EKF prediction step using the motion model.
  :param ekf: ExtendedKalmanFilter instance.
  :param motion_model: Motion model function.
  """
  ekf.predict(lambda state, _: motion_model(state, [0, 0, 0]))

def update(ekf: ExtendedKalmanFilter, player_x, player_y, player_angle, lidar_detections, measurement_model):
  """
  Perform the EKF update step with LiDAR detections.
  :param ekf: ExtendedKalmanFilter instance.
  :param player_x: Robot's current x-coordinate.
  :param player_y: Robot's current y-coordinate.
  :param player_angle: Robot's current orientation in radians.
  :param lidar_detections: LiDAR detections [(distance, angle), ...].
  :param process_lidar_detections: Function to process LiDAR detections into global coordinates.
  :param measurement_model: Measurement model function.
  """
  # Process LiDAR detections
  obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # Match detections with state using Mahalanobis distance
  correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
    obstacle_coordinates, ekf.get_state(), ekf.get_covariance()
  )

  # Update state with correspondences
  measurements = np.copy(ekf.get_state())
  
  # Add robot pose to z
  measurements[0], measurements[1], measurements[2] = player_x, player_y, player_angle
  
  for idx, (pos_x, pos_y) in correspondences:
    measurements[idx + 3], measurements[idx + 1 + 3] = pos_x, pos_y

  # Add unmatched detections as new state dimensions
  for u in unmatched:
    ekf.add_dimension(initial_value=u, process_noise=0.001, measurement_noise=0.01)
    measurements = np.append(measurements, u)

  # Update EKF with the new measurement vector
  z = np.array(measurements).reshape((-1, 1))
  ekf.update(z, measurement_model)

  return correspondences, unmatched
