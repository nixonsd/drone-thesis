import numpy as np

import numpy as np

def mahalanobis_distance(det_x, det_y, obs_x, obs_y, P):
  """
  Compute the Mahalanobis distance between a detection and an obstacle.
  :param det_x: Detected x-coordinate.
  :param det_y: Detected y-coordinate.
  :param obs_x: Obstacle x-coordinate in the state.
  :param obs_y: Obstacle y-coordinate in the state.
  :param P: Covariance matrix (2x2) for the obstacle position.
  :return: Mahalanobis distance.
  """
  try:
    detection = np.array([det_x, det_y]).reshape(-1, 1)
    obstacle = np.array([obs_x, obs_y]).reshape(-1, 1)
    delta = detection - obstacle
    return np.sqrt(delta.T @ np.linalg.inv(P) @ delta)[0, 0]
  except np.linalg.LinAlgError:
    return float('inf')  # Return a high distance if the matrix is singular

def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=5.0):
  """
  Match detections to known obstacles using Mahalanobis distance.
  :param obstacle_coordinates: Flattened array [x1, y1, x2, y2, ...].
  :param state: Current state vector of the EKF [x1, y1, x2, y2, ...].
  :param P: Covariance matrix of the EKF state.
  :param threshold: Mahalanobis distance threshold for valid matches.
  :return: List of correspondences and unmatched detections.
  """
  correspondences = []
  unmatched_detections_set = set()

  # Iterate through obstacle detections
  for i in range(0, len(obstacle_coordinates), 2):
    det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
    closest_idx = None
    min_distance = float('inf')

    # Compare detection with known obstacles in the state
    for j in range(0, state.shape[0], 2):  # Start at index 0, step by 2
      obs_x, obs_y = state[j, 0], state[j + 1, 0]

      # Extract the covariance block for this obstacle
      P_obs = P[j:j + 2, j:j + 2]

      # Handle ill-conditioned covariance matrices
      if np.linalg.cond(P_obs) > 1e12:
        print(f"Warning: Covariance matrix P_obs is ill-conditioned for index {j}")
        continue

      # Compute Mahalanobis distance
      distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)
      
      # Find the closest obstacle within the threshold
      if distance < min_distance and distance <= threshold:
        min_distance = distance
        closest_idx = j // 2  # Index relative to obstacles

    # If a valid match is found, record the correspondence
    if closest_idx is not None:
      correspondences.append((closest_idx, [det_x, det_y]))
    else:
      # Add to unmatched detections (ensure uniqueness)
      unmatched_detections_set.add((det_x, det_y))

  # Convert unmatched set back to list
  unmatched_detections = [list(det) for det in unmatched_detections_set]
  return correspondences, unmatched_detections


# def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=10.0):
#   """
#   Match detections to known obstacles using Mahalanobis distance.
#   :param obstacle_coordinates: Flattened array [x1, y1, x2, y2, ...].
#   :param state: Current state vector of the EKF.
#   :param P: Covariance matrix of the EKF state.
#   :param threshold: Mahalanobis distance threshold for valid matches.
#   :return: List of correspondences and unmatched detections.
#   """
#   correspondences = []
#   unmatched_detections = []

#   # Iterate through obstacle detections
#   for i in range(0, len(obstacle_coordinates), 2):
#     det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
#     closest_idx = None
#     min_distance = float('inf')

#     # Compare with known obstacles in the state
#     for j in range(0, state.shape[0], 2):  # Start at index 0, step by 2
#       obs_x, obs_y = state[j, 0], state[j + 1, 0]

#       # Extract the covariance block for the current obstacle
#       P_obs = P[j:j + 2, j:j + 2]

#       # Compute Mahalanobis distance
#       distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

#       # Update the closest match
#       if distance < min_distance and distance <= threshold:
#         min_distance = distance
#         closest_idx = j // 2  # Index relative to obstacles

#     # If a match is found, add to correspondences
#     if closest_idx is not None:
#       correspondences.append((closest_idx, [det_x, det_y]))
#     else:
#       unmatched_detections.extend([det_x, det_y])

#   return correspondences, unmatched_detections


# def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=3.0):
#   """
#   Match detections to known obstacles using the Mahalanobis distance.
#   :param obstacle_coordinates: Flattened array [x1, y1, x2, y2, ...].
#   :param state: Current state vector of the EKF.
#   :param P: Covariance matrix of the EKF state.
#   :param threshold: Mahalanobis distance threshold for valid matches.
#   :return: List of correspondences and unmatched detections.
#   """
#   correspondences = []
#   unmatched_detections = []

#   # Indices for obstacles in the state
#   obstacle_indices = range(3, state.shape[0], 2)
#   known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

#   # Keep track of which obstacles have been matched
#   matched_indices = set()

#   for i in range(0, len(obstacle_coordinates), 2):
#     det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
#     closest_idx = None
#     min_distance = float('inf')

#     for idx, (obs_x, obs_y) in enumerate(known_obstacles):
#       if idx in matched_indices:
#         continue  # Skip already matched obstacles

#       cov_idx = obstacle_indices[idx]
#       P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]

#       distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

#       if distance < min_distance and distance <= threshold:
#         min_distance = distance
#         closest_idx = idx

#     if closest_idx is not None:
#       correspondences.append((closest_idx, [det_x, det_y]))
#       matched_indices.add(closest_idx)
#     else:
#       unmatched_detections.extend([det_x, det_y])

#   return correspondences, unmatched_detections


# def mahalanobis_distance(det_x, det_y, obs_x, obs_y, P):
#   detection = np.array([det_x, det_y]).reshape(-1, 1)
#   obstacle = np.array([obs_x, obs_y]).reshape(-1, 1)
#   delta = detection - obstacle
#   return np.sqrt(delta.T @ np.linalg.inv(P) @ delta)[0, 0]

# def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=3.0):
#   correspondences = []
#   unmatched_detections = []

#   obstacle_indices = range(3, state.shape[0], 2)
#   known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

#   for i in range(0, len(obstacle_coordinates), 2):
#     det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
#     closest_idx = None
#     min_distance = float('inf')

#     for idx, (obs_x, obs_y) in enumerate(known_obstacles):
#       cov_idx = obstacle_indices[idx]
#       P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]
#       distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

#       if distance < min_distance and distance <= threshold:
#         min_distance = distance
#         closest_idx = idx

#     if closest_idx is not None:
#       correspondences.append((closest_idx, [det_x, det_y]))
#     else:
#       unmatched_detections.extend([det_x, det_y])

#   return correspondences, unmatched_detections
