import numpy as np
import pygame
import math
from settings import SCREEN, PLAYER_SIZE, WHITE, BLUE, RED, GREEN

def draw_uncertainty_circles(state, covariance, scale=10):
  """
  Draw circles around the robot and obstacles to represent uncertainty.
  :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
  :param covariance: EKF covariance matrix.
  :param scale: Factor to scale the circle's radius for better visibility.
  """
  # Draw circle for the robot position (index 0 and 1)
  robot_x, robot_y = state[0, 0], state[1, 0]
  robot_cov = covariance[0:2, 0:2]  # Covariance for robot position
  robot_std_dev = np.sqrt(np.trace(robot_cov))
  robot_radius = int(robot_std_dev * scale)
  pygame.draw.circle(SCREEN, GREEN, (int(robot_x), int(robot_y)), robot_radius, 1)

  # Loop over obstacles (start from index 3)
  for i in range(3, state.shape[0], 2):
    x, y = state[i, 0], state[i + 1, 0]

    # Extract 2x2 covariance block for this obstacle
    P_block = covariance[i:i + 2, i:i + 2]

    # Calculate the standard deviation (radius of the uncertainty circle)
    std_dev = np.sqrt(np.trace(P_block))

    # Scale the radius for better visibility
    radius = int(std_dev * scale)

    # Draw the circle
    pygame.draw.circle(SCREEN, GREEN, (int(x), int(y)), radius, 1)

def draw_player(x, y, angle):
  half_size = PLAYER_SIZE // 2
  corners = [
    (
      x + half_size * math.cos(angle) - half_size * math.sin(angle),
      y + half_size * math.sin(angle) + half_size * math.cos(angle),
    ),
    (
      x - half_size * math.cos(angle) - half_size * math.sin(angle),
      y - half_size * math.sin(angle) + half_size * math.cos(angle),
    ),
    (
      x - half_size * math.cos(angle) + half_size * math.sin(angle),
      y - half_size * math.sin(angle) - half_size * math.cos(angle),
    ),
    (
      x + half_size * math.cos(angle) + half_size * math.sin(angle),
      y + half_size * math.sin(angle) - half_size * math.cos(angle),
    ),
  ]
  pygame.draw.polygon(SCREEN, BLUE, corners)

def draw_obstacles(obstacles):
  for obstacle in obstacles:
    pygame.draw.circle(SCREEN, RED, obstacle, 10)

def draw_lidar_detections(x, y, angle, detections):
  for distance, detection_angle in detections:
    end_x = x + distance * math.cos(angle + detection_angle)
    end_y = y + distance * math.sin(angle + detection_angle)
    pygame.draw.line(SCREEN, GREEN, (x, y), (end_x, end_y), 2)
