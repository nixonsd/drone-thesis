import math

def simulate_lidar(player_x, player_y, player_angle, obstacles, lidar_range, lidar_fov):
  """
  Simulate LiDAR detecting obstacles within range and field of view.
  
  :param player_x: X-coordinate of the player
  :param player_y: Y-coordinate of the player
  :param player_angle: Orientation of the player in radians
  :param obstacles: List of obstacles [(x, y), ...]
  :param lidar_range: Maximum range of the LiDAR
  :param lidar_fov: Field of view of the LiDAR in degrees
  :return: List of detected obstacles as (distance, angle) in radians
  """
  detections = []

  for obstacle in obstacles:
    dx = obstacle[0] - player_x
    dy = obstacle[1] - player_y
    distance = math.sqrt(dx**2 + dy**2)

    # Calculate angle to obstacle in radians
    angle_to_obstacle = math.atan2(dy, dx) - player_angle
    angle_to_obstacle = (angle_to_obstacle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π]

    # Check if the obstacle is within LiDAR range and FOV
    if distance <= lidar_range and abs(angle_to_obstacle) <= math.radians(lidar_fov / 2):
      detections.append((distance, angle_to_obstacle))

  return detections

