import numpy as np
import math
import sympy
import tensorflow as tf
from utils.ekf import find_correspondence_with_mahalanobis_flat
from models.ekf import ExtendedKalmanFilter
from utils.lidar import process_lidar_detections

def offset_left(matrix, offset):
  """
  Offset a matrix to the left by a specified number of columns.
  :param matrix: 2D NumPy array.
  :param offset: Number of columns to shift left.
  :return: New matrix with elements shifted left.
  """
  if offset < 0:
    raise ValueError("Offset must be non-negative.")
  if offset >= matrix.shape[1]:
    return np.zeros_like(matrix)  # All elements are shifted out

  # Create a new matrix with the same shape, initialized to zero
  result = np.zeros_like(matrix)

  # Copy elements to the new matrix with left offset
  result[:, :-offset] = matrix[:, offset:]

  return result

# def motion_model(state, u):
#   """
#   Motion model for EKF.
#   :param state: Current state vector [x, y, theta, ...].
#   :param u: Control input vector [dx, dy, dtheta].
#   :return: Updated state vector.
#   """
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = u

#   if hasattr(x, 'is_symbol') or hasattr(y, 'is_symbol'):
#     # Symbolic computation
#     theta_rad = (theta + dtheta) * sympy.pi / 180
#     x_next = x + dx * sympy.cos(theta_rad)
#     y_next = y + dy * sympy.sin(theta_rad)
#     theta_next = theta + dtheta
#   else:
#     # Numerical computation
#     theta_rad = math.radians(theta + dtheta)
#     x_next = x + dx * math.cos(theta_rad)
#     y_next = y + dy * math.sin(theta_rad)
#     theta_next = theta + dtheta

#   next_state = state.copy()
#   next_state[0, 0], next_state[1, 0], next_state[2, 0] = x_next, y_next, theta_next

#   return next_state

# def motion_model(state, u):
#   """
#   Motion model for EKF.
#   :param state: Current state vector [x, y, theta, ...].
#   :param u: Control input vector [dx, dy, dtheta].
#   :return: Updated state vector.
#   """
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = u

#   # Update state using the control inputs
#   theta_rad = theta + dtheta
#   x_next = x + dx * math.cos(theta_rad)
#   y_next = y + dy * math.sin(theta_rad)
#   theta_next = theta + dtheta

#   # Update and return the new state vector
#   next_state = state.copy()
#   next_state[0, 0], next_state[1, 0], next_state[2, 0] = x_next, y_next, theta_next

#   return next_state

def motion_model(state, control_input):
  """
  Example motion model using NumPy.
  :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
  :param control_input: Control input vector [dx, dy, dtheta].
  :return: Updated state vector.
  """
  # Extract robot state
  x, y, theta = state[0, 0], state[1, 0], state[2, 0]
  dx, dy, dtheta = control_input

  # Update robot state
  theta_next = theta + dtheta
  x_next = x + dx * np.cos(theta_next)
  y_next = y + dy * np.sin(theta_next)

  # Copy and update the state
  updated_state = [x_next, y_next, theta_next]

  # Retain obstacles as-is
  for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
    obs_x, obs_y = state[i, 0], state[i + 1, 0]
    updated_state.extend([obs_x, obs_y])

  return np.array(updated_state).reshape(-1, 1)

def measurement_model(state):
  """
  Generate the measurement prediction vector based on the current state.
  :param state: EKF state vector.
  :return: Measurement vector z_pred.
  """
  # Robot state measurements
  measurements = [state[0, 0], state[1, 0], state[2, 0]]  # x, y, theta

  # Append obstacle positions
  for i in range(3, state.shape[0], 2):
    obs_x, obs_y = state[i, 0], state[i + 1, 0]
    measurements.extend([obs_x, obs_y])

  return np.array(measurements).reshape(-1, 1)


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
#     obs_x, obs_y = state[i, 0], state[i + 1, 0]
#     measurements.extend([obs_x, obs_y])

#   return np.array(measurements).reshape((-1, 1))

# def predict(ekf: ExtendedKalmanFilter, motion_model):
#   """
#   Perform the EKF prediction step using the motion model.
#   :param ekf: ExtendedKalmanFilter instance.
#   :param motion_model: Motion model function.
#   """
#   ekf.predict(lambda state, _: motion_model(state, [0, 0, 0]))

def predict(ekf: ExtendedKalmanFilter, motion_model, control_input):
  """
  Perform the EKF prediction step using the motion model.
  :param ekf: ExtendedKalmanFilter instance.
  :param motion_model: Motion model function.
  :param control_input: Control input vector.
  """
  ekf.predict(motion_model, control_input)

def update(ekf: ExtendedKalmanFilter, player_x, player_y, player_angle, lidar_detections, measurement_model):
  """
  Perform the EKF update step with LiDAR detections.
  :param ekf: ExtendedKalmanFilter instance.
  :param player_x: Robot's current x-coordinate.
  :param player_y: Robot's current y-coordinate.
  :param player_angle: Robot's current orientation in radians.
  :param lidar_detections: LiDAR detections [(distance, angle), ...].
  :param measurement_model: Measurement model function.
  """
  # Process LiDAR detections
  obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

  # Match detections with state using Mahalanobis distance
  correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
    obstacle_coordinates, ekf.get_state()[3:], offset_left(ekf.get_covariance()[3:, :], 3)
  )

  # Update state with correspondences
  measurements = np.copy(ekf.get_state())

  # Add robot pose to z
  measurements[0], measurements[1], measurements[2] = player_x, player_y, player_angle

  # Update correspondences in the state
  for idx, (pos_x, pos_y) in correspondences:
    measurements[3 + idx * 2] = pos_x
    measurements[3 + idx * 2 + 1] = pos_y

  # Ensure unmatched detections are unique
  unmatched_set = set(tuple(u) for u in np.array(unmatched).reshape(-1, 2))

  # Add unmatched detections as new state dimensions
  for det_x, det_y in unmatched_set:
    # Check if the unmatched detection is already in the state
    existing_obstacles = ekf.get_state()[3:].reshape(-1, 2)
    if not np.any((existing_obstacles == [det_x, det_y]).all(axis=1)):
      ekf.add_dimension(initial_value=[det_x, det_y], process_noise=3, measurement_noise=3)
      measurements = np.append(measurements, [det_x, det_y])

  # Update EKF with the new measurement vector
  z = measurements.reshape((-1, 1))
  ekf.update(z, measurement_model)
  
  return correspondences, unmatched

