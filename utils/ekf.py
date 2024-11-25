import numpy as np

def mahalanobis_distance(det_x, det_y, obs_x, obs_y, P):
  detection = np.array([det_x, det_y]).reshape(-1, 1)
  obstacle = np.array([obs_x, obs_y]).reshape(-1, 1)
  delta = detection - obstacle
  return np.sqrt(delta.T @ np.linalg.inv(P) @ delta)[0, 0]

def find_correspondence_with_mahalanobis_flat(obstacle_coordinates, state, P, threshold=3.0):
  correspondences = []
  unmatched_detections = []

  obstacle_indices = range(3, state.shape[0], 2)
  known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

  for i in range(0, len(obstacle_coordinates), 2):
    det_x, det_y = obstacle_coordinates[i], obstacle_coordinates[i + 1]
    closest_idx = None
    min_distance = float('inf')

    for idx, (obs_x, obs_y) in enumerate(known_obstacles):
      cov_idx = obstacle_indices[idx]
      P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]
      distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

      if distance < min_distance and distance <= threshold:
        min_distance = distance
        closest_idx = idx

    if closest_idx is not None:
      correspondences.append((closest_idx, [det_x, det_y]))
    else:
      unmatched_detections.extend([det_x, det_y])

  return correspondences, unmatched_detections
