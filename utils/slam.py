import numpy as np
import tensorflow as tf
from settings import DETECTION_THRESHOLD, LIDAR_MEASUREMENT_NOISE, LIDAR_PROCESS_NOISE
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

# def motion_model(state, control_input):
#   """
#   Example motion model using NumPy.
#   :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
#   :param control_input: Control input vector [dx, dy, dtheta].
#   :return: Updated state vector.
#   """
#   # Extract robot state
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = control_input

#   # Update robot state
#   theta_next = theta + dtheta
#   x_next = x + dx
#   y_next = y + dy

#   # Copy and update the state
#   updated_state = [x_next, y_next, theta_next]

#   # Retain obstacles as-is
#   for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
#     obs_x, obs_y = state[i, 0], state[i + 1, 0]
#     updated_state.extend([obs_x, obs_y])

#   return np.array(updated_state).reshape(-1, 1)

def motion_model(state, control_input, dt):
    """
    Motion model for the EKF.
    :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
    :param control_input: Control input vector [v_linear, v_angular].
    :param dt: Time step for velocity integration.
    :return: Updated state vector.
    """
    x, y, theta = state[0, 0], state[1, 0], state[2, 0]
    v_linear, v_angular = control_input

    # Update the robot's position and orientation using the motion model
    theta_next = theta + v_angular * dt
    theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
    x_next = x + v_linear * np.cos(theta_next) * dt
    y_next = y + v_linear * np.sin(theta_next) * dt

    # Copy and update the state
    updated_state = [x_next, y_next, theta_next]

    # Retain obstacles as-is
    for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
        obs_x, obs_y = state[i, 0], state[i + 1, 0]
        updated_state.extend([obs_x, obs_y])

    return np.array(updated_state).reshape(-1, 1)

# def motion_model(state, control_input, dt, alpha=0.95):
#     """
#     Motion model with a weighted update to mitigate errors.
#     :param alpha: Weight factor for state update (default 0.95).
#     """
#     x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#     v_linear, v_angular = control_input

#     # Predict new orientation
#     theta_next = (theta + v_angular * dt + np.pi) % (2 * np.pi) - np.pi

#     # Predict new position
#     x_next = x + alpha * (v_linear * np.cos(theta_next) * dt)
#     y_next = y + alpha * (v_linear * np.sin(theta_next) * dt)

#     # Retain obstacle positions
#     updated_state = [x_next, y_next, theta_next]
#     for i in range(3, state.shape[0], 2):
#         updated_state.extend([state[i, 0], state[i + 1, 0]])

#     return np.array(updated_state).reshape(-1, 1)

# def motion_model(state, control_input, dt):
#   """
#   Example motion model using NumPy.
#   :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
#   :param control_input: Control input vector [dx, dy, dtheta].
#   :return: Updated state vector.
#   """
#   # Extract robot state
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = control_input

#   # Update robot state
#   theta_next = round(theta + dtheta, 8)
#   x_next = round(x + dx, 4)
#   y_next = round(y + dy, 4)

#   # Copy and update the state
#   updated_state = [x_next, y_next, theta_next]

#   # Retain obstacles as-is
#   for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
#     obs_x, obs_y = state[i, 0], state[i + 1, 0]
#     updated_state.extend([obs_x, obs_y])

#   return np.array(updated_state).reshape(-1, 1)


# def motion_model(state, control_input, dt):
#   """
#   Example motion model using NumPy.
#   :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
#   :param control_input: Control input vector [dx, dy, dtheta].
#   :return: Updated state vector.
#   """
#   # Extract robot state
#   x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#   dx, dy, dtheta = control_input

#   # Update robot state
#   theta_next = theta + dtheta
#   x_next = x + dx
#   y_next = y + dy

#   # Copy and update the state
#   updated_state = [x_next, y_next, theta_next]

#   # Retain obstacles as-is
#   for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
#     obs_x, obs_y = state[i, 0], state[i + 1, 0]
#     updated_state.extend([obs_x, obs_y])

#   return np.array(updated_state).reshape(-1, 1)

# def measurement_model(state, measurements, dt):
#   """
#   Generate the measurement prediction vector based on the current state.
#   :param state: EKF state vector.
#   :return: Measurement vector z_pred.
#   """
#   # Robot state measurements
#   print(measurements)
#   _measurements = [measurements[0, 0], measurements[1, 0], measurements[2, 0]]  # x_speed, y_speed, theta_speed
#   _measurements = state + _measurements * dt # overall_x, overall_y, overall_theta
  
#   # Append obstacle positions
#   for i in range(3, state.shape[0], 2):
#     obs_x, obs_y = state[i, 0], state[i + 1, 0]
#     _measurements.extend([obs_x, obs_y])

#   return np.array(_measurements).reshape(-1, 1)

# def motion_model(state, control_input, dt):
#     """
#     Motion model for the EKF.
#     :param state: Current state vector [x, y, theta, obs1_x, obs1_y, ...].
#     :param control_input: Control input vector [v_linear, v_angular].
#     :param dt: Time step for velocity integration.
#     :return: Updated state vector.
#     """
#     x, y, theta = state[0, 0], state[1, 0], state[2, 0]
#     v_linear, v_angular = control_input

#     # Update the robot's position and orientation using the motion model
#     theta_next = theta + v_angular * dt
#     x_next = x + v_linear * np.cos(theta_next) * dt
#     y_next = y + v_linear * np.sin(theta_next) * dt

#     # Copy and update the state
#     updated_state = [x_next, y_next, theta_next]

#     # Retain obstacles as-is
#     for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
#         obs_x, obs_y = state[i, 0], state[i + 1, 0]
#         updated_state.extend([obs_x, obs_y])

#     return np.array(updated_state).reshape(-1, 1)

def measurement_model(state, dt):
    """
    Generate the measurement prediction vector based on the current state.
    :param state: EKF state vector [x, y, theta, obs1_x, obs1_y, ...].
    :param dt: Time step for measurement prediction.
    :return: Measurement vector z_pred.
    """
    # Robot's position and orientation
    x, y, theta = state[0], state[1], state[2]

    # Predict measurements
    z_pred = np.zeros(np.shape(state))
    z_pred[:3] = [x, y, theta]

    # Add obstacle positions to predicted measurements
    for i in range(3, state.shape[0], 2):  # Obstacles are [x, y] pairs
        obs_x, obs_y = state[i], state[i + 1]

        # Compute predicted range and bearing to each obstacle
        r = np.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
        bearing = (np.arctan2(obs_y - y, obs_x - x) - theta + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

        z_pred[i] = r
        z_pred[i + 1] = bearing

    return z_pred.reshape(-1, 1)

def pad_to_shape(state, target_shape):
    """
    Pads the state vector with zeros to match the desired shape.
    :param state: Current state vector (numpy array).
    :param target_shape: Desired shape (int, tuple, or list).
    :return: Padded state vector with the desired shape.
    """
    current_shape = state.shape[0]
    
    # Calculate the number of zeros needed
    if isinstance(target_shape, (tuple, list)):
        target_size = target_shape[0]
    else:
        target_size = target_shape

    if current_shape >= target_size:
        return state  # No padding needed
    
    # Create a zero array for padding
    padding_size = target_size - current_shape
    padding = np.zeros((padding_size, state.shape[1]))
    
    # Append the padding to the state vector
    padded_state = np.vstack((state, padding))
    return padded_state


# def measurement_model(ekf: ExtendedKalmanFilter, previous_state, measurements, dt, P, threshold=5.0):
#   """
#   Predict the measurement vector based on the robot's current state, measurements, and correspondences.
#   :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
#   :param measurements: [x_velocity, y_velocity, theta_velocity, r1, theta1, r2, theta2, ...].
#                        Includes velocities and LiDAR obstacle readings.
#   :param dt: Time step for velocity integration.
#   :param P: Covariance matrix of the EKF state.
#   :param threshold: Mahalanobis distance threshold for valid matches.
#   :return: Predicted measurement vector z_pred.
#   """
#   # Initialize predicted measurement vector with robot state
#   z_pred = np.zeros(np.shape(previous_state))
#   z_pred[:3] = predicted_x, predicted_y, predicted_angle
    
#   return np.array(z_pred).reshape(-1, 1)

def predict(ekf: ExtendedKalmanFilter, control_input, dt):
    """
    EKF prediction step.
    :param ekf: ExtendedKalmanFilter instance.
    :param control_input: Control input [v_linear, v_angular].
    :param dt: Time step.
    """
    ekf.predict(lambda state: motion_model(state, control_input, dt))

#def update(ekf: ExtendedKalmanFilter, measurements, measurement_model, dt, threshold=5):
# def update(ekf: ExtendedKalmanFilter, measurements, dt, threshold=5.0):
#   """
#   EKF update step.
#   :param ekf: ExtendedKalmanFilter instance.
#   :param measurements: Actual measurements [x_velocity, y_velocity, theta_velocity, r1, theta1, ...].
#   :param dt: Time step.
#   :param threshold: Mahalanobis distance threshold for valid matches.
#   """
#   predicted_state = ekf.get_state()
#   P = ekf.get_covariance()
  
#   # Extract robot state
#   robot_x, robot_y, robot_angle = predicted_state[0], predicted_state[1], predicted_state[2]

#   # Update robot state using velocities
#   x_velocity, y_velocity, theta_velocity = measurements[0], measurements[1], measurements[2]
#   predicted_x = robot_x + x_velocity * dt
#   predicted_y = robot_y + y_velocity * dt
#   predicted_angle = (robot_angle + theta_velocity * dt + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

#   # Initialize predicted measurement vector with robot state
#   z = np.zeros(np.shape(predicted_state))
#   z[:3] = predicted_x, predicted_y, predicted_angle

#   # Convert obstacle measurements (r, theta) to (x, y) in global coordinates
#   obstacle_coordinates = []
#   num_obstacles = (len(measurements) - 3) // 2
#   for i in range(num_obstacles):
#     r = measurements[3 + 2 * i]  # Distance to obstacle
#     theta = measurements[4 + 2 * i]  # Angle to obstacle (relative to robot)

#     # Calculate obstacle global coordinates
#     obs_x = predicted_x + r * np.cos(predicted_angle + theta)
#     obs_y = predicted_y + r * np.sin(predicted_angle + theta)
#     obstacle_coordinates.extend([obs_x, obs_y])

#   # Match obstacle detections with known obstacles in the state
#   correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
#     np.array(obstacle_coordinates), predicted_state[3:], P[3:, 3:], threshold
#   )

#   # Update correspondences in the state
#   for idx, (pos_x, pos_y) in correspondences:
#     z[3 + idx * 2] = pos_x
#     z[3 + idx * 2 + 1] = pos_y

#   # Add unmatched obstacles to the predicted measurement vector
#   for unmatched_obs in unmatched:
#     z = np.append(z, unmatched_obs)
#     ekf.add_dimension(initial_value=unmatched_obs, process_noise=0.001, measurement_noise=0.1)
  
#   predicted_state = pad_to_shape(predicted_state, np.shape(z))
  
#   # Perform EKF update
#   ekf.update(z, lambda state: predicted_state)

def update(ekf: ExtendedKalmanFilter, control_input, measurements, dt, threshold=5.0):
    # Predict step
    ekf.predict(lambda state: motion_model(state, control_input, dt))

    # Extract predicted state and covariance
    predicted_state = ekf.get_state()
    print('predicted_state', predicted_state)
    P = ekf.get_covariance()

    # Initialize 1D predicted measurement vector z_pred
    z_pred = np.zeros(predicted_state.shape[0])
    z_pred[:3] = predicted_state[:3, 0]

    # Initialize 1D measurement vector z
    z = np.zeros(predicted_state.shape[0])
    
    robot_x, robot_y, robot_angle = z_pred[0], z_pred[1], z_pred[2]
    x_velocity, y_velocity, theta_velocity = measurements[0], measurements[1], measurements[2]
    measured_angle = robot_angle + theta_velocity * dt
    measured_angle = (measured_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
    measured_x = robot_x + x_velocity * dt
    measured_y = robot_y + y_velocity * dt

    z[:3] = [measured_x, measured_y, measured_angle]

    # # Predicted measurement vector (z_pred)
    # z_pred = np.zeros_like(predicted_state)
    # z_pred[:3] = motion_model(predicted_state, control_input, dt)

    # print(z_pred)

    
    # # Measurement vector (z)
    # z = np.zeros_like(predicted_state)
    # print('predicted_state = ', predicted_state,  'z = ', z, 'measurements = ', measurements)
    # z[:3,:1] = measurements[:3] # Direct robot state measurements
    # print('predicted_state = ', predicted_state,  'z = ', z, 'measurements = ', measurements)

    # # # Handle obstacle measurements
    # obstacle_coordinates = []
    # num_obstacles = (len(measurements) - 3) // 2
    # for i in range(num_obstacles):
    #     r, theta = measurements[3 + 2 * i], measurements[3 + 2 * i + 1]
    #     obs_x = z[0] + r * np.cos(z[2] + theta)
    #     obs_y = z[1] + r * np.sin(z[2] + theta)
    #     obstacle_coordinates.extend([obs_x, obs_y])

    # # Match detected obstacles to state obstacles
    # correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
    #     np.array(obstacle_coordinates), predicted_state[3:], P[3:, 3:], threshold
    # )
    
    # Convert obstacle measurements (r, theta) to (x, y) in global coordinates
    obstacle_coordinates = []
    num_obstacles = (len(measurements) - 3) // 2
    for i in range(num_obstacles):
        r, theta = measurements[3 + 2 * i], measurements[4 + 2 * i]  # Angle to obstacle (relative to robot)
        obs_x = z[0] + r * np.cos(z[2] + theta)
        obs_y = z[1] + r * np.sin(z[2] + theta)
        obstacle_coordinates.extend([obs_x, obs_y])

    # Match obstacle detections with known obstacles in the state
    correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
        np.array(obstacle_coordinates), predicted_state[3:], P[3:, 3:], threshold
    )

    for idx, (obs_x, obs_y) in correspondences:
        z[3 + idx * 2] = obs_x
        z[3 + idx * 2 + 1] = obs_y
        z_pred[3 + idx * 2] = obs_x
        z_pred[3 + idx * 2 + 1] = obs_y

    # Append unmatched obstacles to state and covariance matrix
    for unmatched_obs in unmatched:
        z = np.append(z, unmatched_obs)
        z_pred = np.append(z_pred, unmatched_obs)
        ekf.add_dimension(
            initial_value=unmatched_obs,
            process_noise=LIDAR_PROCESS_NOISE[1],
            measurement_noise=LIDAR_MEASUREMENT_NOISE
        )

    print('angles', np.degrees(z[2]), np.degrees(z_pred[2]))

    # Update step
    ekf.update(z.reshape(-1, 1), lambda state: z_pred.reshape(-1, 1))
    print('x', np.degrees(ekf.x[2]))

# def update(ekf: ExtendedKalmanFilter, control_input, measurements, dt, threshold=5.0):
#     """
#     EKF update step with control input for predicted path.
#     :param ekf: ExtendedKalmanFilter instance.
#     :param control_input: Control input vector [v_linear, v_angular].
#     :param measurements: Actual measurements [x_velocity, y_velocity, theta_velocity, r1, theta1, ...].
#     :param dt: Time step.
#     :param threshold: Mahalanobis distance threshold for valid matches.
#     """
#     # Get the current predicted state and covariance matrix
#     predicted_state = ekf.get_state()
#     P = ekf.get_covariance()

#     # --- Compute predicted path (z_pred) based on control input ---
#     # Extract robot state
#     robot_x, robot_y, robot_angle = predicted_state[0], predicted_state[1], predicted_state[2]
#     v_linear, v_angular = control_input

#     # Update robot position using control input
#     predicted_angle = robot_angle + v_angular * dt
#     predicted_angle = (predicted_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
#     predicted_x = robot_x + v_linear * np.cos(predicted_angle) * dt
#     predicted_y = robot_y + v_linear * np.sin(predicted_angle) * dt

#     # Initialize predicted measurement vector (z_pred)
#     z_pred = np.zeros_like(predicted_state)
#     z_pred[:3] = [predicted_x, predicted_y, predicted_angle]

#     # Retain obstacles in predicted state
#     for i in range(3, predicted_state.shape[0], 2):
#         z_pred[i] = predicted_state[i, 0]
#         z_pred[i + 1] = predicted_state[i + 1, 0]

#     # --- Compute measured path (z) based on measurements ---
#     x_velocity, y_velocity, theta_velocity = measurements[0], measurements[1], measurements[2]
#     measured_angle = robot_angle + theta_velocity * dt
#     measured_angle = (measured_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
#     measured_x = robot_x + x_velocity * dt
#     measured_y = robot_y + y_velocity * dt

#     # Initialize actual measurement vector (z)
#     z = np.zeros_like(predicted_state)
#     z[:3] = [measured_x, measured_y, measured_angle]

#     # Convert obstacle measurements (r, theta) to (x, y) in global coordinates
#     obstacle_coordinates = []
#     num_obstacles = (len(measurements) - 3) // 2
#     for i in range(num_obstacles):
#         r = measurements[3 + 2 * i]  # Distance to obstacle
#         theta = measurements[4 + 2 * i]  # Angle to obstacle (relative to robot)
#         obs_x = measured_x + r * np.cos(measured_angle + theta)
#         obs_y = measured_y + r * np.sin(measured_angle + theta)
#         obstacle_coordinates.extend([obs_x, obs_y])

#     # Match obstacle detections with known obstacles in the state
#     correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
#         np.array(obstacle_coordinates), predicted_state[3:], P[3:, 3:], threshold
#     )

#     # Update correspondences in the state
#     for idx, (pos_x, pos_y) in correspondences:
#         z[3 + idx * 2] = pos_x
#         z[3 + idx * 2 + 1] = pos_y

#     # Add unmatched obstacles to the state
#     for unmatched_obs in unmatched:
#         z = np.append(z, unmatched_obs)
#         z_pred = np.append(z_pred, unmatched_obs)
#         ekf.add_dimension(initial_value=unmatched_obs, process_noise=0.01, measurement_noise=[0.3, 0.1])

#     # Perform the EKF update
#     ekf.update(z, lambda state: z_pred)


# def update(ekf: ExtendedKalmanFilter, player_x, player_y, player_angle, lidar_detections, measurement_model):
#   """
#   Perform the EKF update step with LiDAR detections.
#   :param ekf: ExtendedKalmanFilter instance.
#   :param player_x: Robot's current x-coordinate.
#   :param player_y: Robot's current y-coordinate.
#   :param player_angle: Robot's current orientation in radians.
#   :param lidar_detections: LiDAR detections [(distance, angle), ...].
#   :param measurement_model: Measurement model function.
#   """
#   # Process LiDAR detections
#   obstacle_coordinates = process_lidar_detections(player_x, player_y, player_angle, lidar_detections)

#   # Match detections with state using Mahalanobis distance
#   correspondences, unmatched = find_correspondence_with_mahalanobis_flat(
#     obstacle_coordinates, ekf.get_state()[3:], offset_left(ekf.get_covariance()[3:, :], 3), 20
#   )

#   # Update state with correspondences
#   measurements = np.copy(ekf.get_state())

#   # Add robot pose to z
#   measurements[0], measurements[1], measurements[2] = player_x, player_y, player_angle

#   # Update correspondences in the state
#   for idx, (pos_x, pos_y) in correspondences:
#     measurements[3 + idx * 2] = pos_x
#     measurements[3 + idx * 2 + 1] = pos_y

#   # Ensure unmatched detections are unique
#   unmatched_set = set(tuple(u) for u in np.array(unmatched).reshape(-1, 2))

#   # Add unmatched detections as new state dimensions
#   for det_x, det_y in unmatched_set:
#     # Check if the unmatched detection is already in the state
#     existing_obstacles = ekf.get_state()[3:].reshape(-1, 2)
#     if not np.any((existing_obstacles == [det_x, det_y]).all(axis=1)):
#       ekf.add_dimension(initial_value=[det_x, det_y], process_noise=3, measurement_noise=3)
#       measurements = np.append(measurements, [det_x, det_y])

#   # Update EKF with the new measurement vector
#   z = measurements.reshape((-1, 1))
#   ekf.update(z, measurement_model)
  
#   return correspondences, unmatched
