import pygame
import numpy as np
from settings import *
from utils.render import draw_player, draw_obstacles, draw_lidar_detections, draw_text, draw_uncertainty_circles
from models.lidar_simulator import simulate_lidar
from models.ekf import ExtendedKalmanFilter
from utils.slam import measurement_model, motion_model, predict, update
from utils.noise import add_noise

# EKF setup
initial_state_dim = 3  # [player_x, player_y, player_angle]
ekf = ExtendedKalmanFilter(state_dim=initial_state_dim, meas_dim=initial_state_dim)
ekf.set_process_noise(np.diag([0.1] * initial_state_dim))  # Process noise
ekf.set_measurement_noise(np.diag([1.0] * initial_state_dim))  # Measurement noise

# Initialize EKF state to unknown (zeros)
ekf.x = np.zeros((initial_state_dim, 1))

# Player actual position (ground truth)
actual_player_x, actual_player_y, actual_player_angle = 50, 50, 0

# Pygame setup
pygame.init()
clock = pygame.time.Clock()
running = True

# Game loop
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  # Previous position (for control input calculation)
  prev_x, prev_y, prev_angle = actual_player_x, actual_player_y, actual_player_angle

  # Movement keys
  keys = pygame.key.get_pressed()
  if keys[pygame.K_UP]:
    actual_player_x += PLAYER_SPEED * np.cos(actual_player_angle)
    actual_player_y += PLAYER_SPEED * np.sin(actual_player_angle)
  if keys[pygame.K_DOWN]:
    actual_player_x -= PLAYER_SPEED * np.cos(actual_player_angle)
    actual_player_y -= PLAYER_SPEED * np.sin(actual_player_angle)
  if keys[pygame.K_LEFT]:
    actual_player_angle -= np.radians(ROTATION_SPEED)
  if keys[pygame.K_RIGHT]:
    actual_player_angle += np.radians(ROTATION_SPEED)

  # Normalize angle
  actual_player_angle = (actual_player_angle + np.pi) % (2 * np.pi) - np.pi

  # Add noise to the ground truth to simulate noisy measurements
  noisy_player_x = add_noise(actual_player_x, 3)
  noisy_player_y = add_noise(actual_player_y, 3)
  noisy_player_angle = add_noise(actual_player_angle, 0.1)

  # Simulate noisy LiDAR detections
  lidar_detections = simulate_lidar(
    actual_player_x, actual_player_y, actual_player_angle, OBSTACLES, LIDAR_RANGE, LIDAR_FOV
  )
  noisy_lidar_detections = [(add_noise(r, 0.1), theta) for r, theta in lidar_detections]

  # SLAM Prediction Step: Predict player's position using control input
  dx = noisy_player_x - prev_x
  dy = noisy_player_y - prev_y
  dtheta = noisy_player_angle - prev_angle
  control_input = [dx, dy, dtheta]
  predict(ekf, motion_model, control_input)

  # SLAM Update Step: Refine player's position using noisy measurements
  correspondences, unmatched = update(
    ekf, noisy_player_x, noisy_player_y, noisy_player_angle, noisy_lidar_detections, measurement_model
  )

  # Extract estimated player position from EKF state
  estimated_player_x, estimated_player_y, estimated_player_angle = ekf.get_state().flatten()[:3]

  # Render frame
  SCREEN.fill(WHITE)

  # Draw actual player
  draw_player(actual_player_x, actual_player_y, actual_player_angle)
  
  # Draw obstacles and uncertainty circles
  draw_obstacles(OBSTACLES)
  draw_uncertainty_circles(ekf.get_state(), ekf.get_covariance())

  # Draw LiDAR detections
  draw_lidar_detections(actual_player_x, actual_player_y, actual_player_angle, noisy_lidar_detections)

  # Display actual, noisy, and estimated positions
  actual_text = f"Actual: x={actual_player_x:.2f}, y={actual_player_y:.2f}, angle={np.degrees(actual_player_angle):.2f}"
  # noisy_text = f"Noisy: x={noisy_player_x:.2f}, y={noisy_player_y:.2f}, angle={np.degrees(noisy_player_angle):.2f}"
  estimated_text = f"Estimated: x={estimated_player_x:.2f}, y={estimated_player_y:.2f}, angle={np.degrees(estimated_player_angle):.2f}"
  draw_text(SCREEN, actual_text, position=(10, 10), font_size=24, color=RED)
  # draw_text(SCREEN, noisy_text, position=(10, 40), font_size=24, color=BLUE)
  draw_text(SCREEN, estimated_text, position=(10, 70), font_size=24, color=BLUE)

  pygame.display.flip()
  clock.tick(FPS)

pygame.quit()


# import pygame
# import numpy as np
# from settings import *
# from utils.render import draw_player, draw_obstacles, draw_lidar_detections, draw_text, draw_uncertainty_circles
# from models.lidar_simulator import simulate_lidar
# from models.ekf import ExtendedKalmanFilter
# from utils.slam import measurement_model, motion_model, predict, update
# from utils.noise import add_noise

# # EKF setup
# ekf = ExtendedKalmanFilter(state_dim=3, meas_dim=3)
# ekf.set_process_noise(np.diag([0.1] * 3))
# ekf.set_measurement_noise(np.diag([1.0] * 3))

# # player_x, player_y, player_angle = 0, 0, 0

# # Player actual position (ground truth)
# actual_player_x, actual_player_y, actual_player_angle = 50, 50, 0

# # Player estimated position (SLAM output)
# estimated_player_x, estimated_player_y, estimated_player_angle = 0, 0, 0

# clock = pygame.time.Clock()

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Previous position and angle (for control input calculation)
#     prev_x, prev_y, prev_angle = actual_player_x, actual_player_y, actual_player_angle

#     # Movement keys
#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_UP]:
#       actual_player_x += PLAYER_SPEED * np.cos(actual_player_angle)
#       actual_player_y += PLAYER_SPEED * np.sin(actual_player_angle)
#     if keys[pygame.K_DOWN]:
#       actual_player_x -= PLAYER_SPEED * np.cos(actual_player_angle)
#       actual_player_y -= PLAYER_SPEED * np.sin(actual_player_angle)
#     if keys[pygame.K_LEFT]:
#       actual_player_angle -= np.radians(ROTATION_SPEED)
#     if keys[pygame.K_RIGHT]:
#       actual_player_angle += np.radians(ROTATION_SPEED)

#     actual_player_angle = (actual_player_angle + np.pi) % (2 * np.pi) - np.pi

#     # Simulate LiDAR detections (ground truth position is used here)
#     lidar_detections = simulate_lidar(
#       actual_player_x, actual_player_y, actual_player_angle, OBSTACLES, LIDAR_RANGE, LIDAR_FOV
#     )
    
#     # Add noise to player position and angle (simulate noisy measurements)
#     noisy_player_x = add_noise(actual_player_x, sigma=15)
#     noisy_player_y = add_noise(actual_player_y, sigma=15)
#     noisy_player_angle = add_noise(actual_player_angle, sigma=0.5)
    
#     # Calculate real control input
#     dx = noisy_player_x - prev_x
#     dy = noisy_player_y - prev_y
#     dtheta = noisy_player_angle - prev_angle
#     control_input = [dx, dy, dtheta]
    
#     # SLAM Prediction Step
#     predict(ekf, motion_model, control_input)

#     # SLAM Update Step
#     correspondences, unmatched = update(
#       ekf, actual_player_x, actual_player_y, actual_player_angle, lidar_detections, measurement_model
#     )
  
#     # Render frame
#     SCREEN.fill(WHITE)

#     # Draw player and obstacles
#     draw_player(actual_player_x, actual_player_y, actual_player_angle)
#     draw_obstacles(OBSTACLES)

#     # Draw uncertainty circles for robot and obstacles (excluding robot angle)
#     draw_uncertainty_circles(ekf.get_state(), ekf.get_covariance())

#     # Draw LiDAR detections
#     draw_lidar_detections(actual_player_x, actual_player_y, actual_player_angle, lidar_detections)
    
#     # Extract estimated player position from EKF state
#     estimated_player_x, estimated_player_y, estimated_player_angle = ekf.get_state().flatten()[:3]
    
#     # Display actual and estimated positions
#     actual_text = f"Actual: x={actual_player_x:.2f}, y={actual_player_y:.2f}, angle={np.degrees(actual_player_angle):.2f}"
#     estimated_text = f"Estimated: x={estimated_player_x:.2f}, y={estimated_player_y:.2f}, angle={np.degrees(estimated_player_angle):.2f}"
#     draw_text(SCREEN, actual_text, position=(10, 10), font_size=24, color=RED)
#     draw_text(SCREEN, estimated_text, position=(10, 40), font_size=24, color=GREEN)
    

#     pygame.display.flip()
#     clock.tick(FPS)

# pygame.quit()
