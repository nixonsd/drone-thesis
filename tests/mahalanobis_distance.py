import numpy as np

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

def find_correspondences_with_mahalanobis(obstacle_coordinates, state, P, threshold=3.0):
    """
    Finds correspondences between detections and obstacles using Mahalanobis distance.
    :param obstacle_coordinates: List of detected obstacle coordinates [(x1, y1), (x2, y2), ...].
    :param state: Current state vector of the EKF (includes robot and obstacle states).
    :param P: Full covariance matrix of the state vector.
    :param threshold: Mahalanobis distance threshold for valid matches.
    :return: List of correspondences and unmatched detections.
    """
    correspondences = []
    unmatched_detections = []

    obstacle_indices = range(3, state.shape[0], 2)  # Indices for obstacles in the state
    known_obstacles = [(state[i, 0], state[i + 1, 0]) for i in obstacle_indices]

    for det_x, det_y in obstacle_coordinates:
        closest_idx = None
        min_distance = float('inf')

        for idx, (obs_x, obs_y) in enumerate(known_obstacles):
            # Extract the covariance matrix for this obstacle (2x2 block)
            cov_idx = obstacle_indices[idx]
            P_obs = P[cov_idx:cov_idx + 2, cov_idx:cov_idx + 2]

            # Compute Mahalanobis distance
            distance = mahalanobis_distance(det_x, det_y, obs_x, obs_y, P_obs)

            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_idx = idx

        if closest_idx is not None:
            correspondences.append((closest_idx, (det_x, det_y)))
        else:
            unmatched_detections.append((det_x, det_y))

    return correspondences, unmatched_detections

# Example Usage
state = np.array([
    [100.0], [200.0], [0.0],  # Robot x, y, theta
    [300.0], [400.0],         # Obstacle 1 x, y
    [500.0], [600.0]          # Obstacle 2 x, y
])

P = np.eye(state.shape[0]) * 10  # Full state covariance matrix
lidar_detections = [(310, 390), (700, 800), (300, 400)]  # Simulated LiDAR detections

correspondences, unmatched = find_correspondences_with_mahalanobis(lidar_detections, state, P)

print("Correspondences:")
for obs_idx, (det_x, det_y) in correspondences:
    print(f"Obstacle {obs_idx} corresponds to detected coordinates ({det_x}, {det_y})")

print("Unmatched Detections:")
for det_x, det_y in unmatched:
    print(f"New obstacle detected at ({det_x}, {det_y})")
