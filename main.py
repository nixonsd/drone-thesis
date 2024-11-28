import pygame
import numpy as np
from settings import *
from utils.render import draw_estimated_positions, draw_player, draw_obstacles, draw_lidar_detections, draw_text, draw_uncertainty_ellipses
from models.lidar_simulator import simulate_lidar
from models.ekf import ExtendedKalmanFilter
from utils.slam import predict, update
from utils.noise import add_noise

# Player actual position (ground truth)
actual_player_x, actual_player_y, actual_player_angle = 50, 50, 0

# Player estimated position
estimated_player_x, estimated_player_y, estimated_player_angle = 50, 50, 0

# EKF setup
initial_state_dim = 3  # [player_x, player_y, player_angle]
ekf = ExtendedKalmanFilter(state_dim=initial_state_dim, meas_dim=initial_state_dim)
ekf.set_process_noise(np.eye(3) * PLAYER_PROCESS_NOISE)  # Process noise
ekf.set_measurement_noise(np.eye(3) * [*ENCODER_MEASUREMENT_NOISE, IMU_NOISE[2]])  # Measurement noise

# Initialize EKF state to unknown (zeros)
ekf.x = np.reshape([
  estimated_player_x,
  estimated_player_y,
  estimated_player_angle
], (3, 1))

# Pygame setup
pygame.init()
clock = pygame.time.Clock()
running = True
dt = 1 / FPS

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

  # Simulate noisy LiDAR detections
  lidar_detections = simulate_lidar(
    actual_player_x, actual_player_y, actual_player_angle, OBSTACLES, LIDAR_RANGE, LIDAR_FOV
  )
  # noisy_lidar_detections = [(add_noise(r, 0.1), theta) for r, theta in lidar_detections]
  noisy_lidar_detections = [(add_noise(r, LIDAR_MEASUREMENT_NOISE[0]), add_noise(theta, LIDAR_MEASUREMENT_NOISE[1])) for r, theta in lidar_detections]
  
  # SLAM Prediction Step: Predict player's position using control input
  dx = actual_player_x - prev_x
  dy = actual_player_y - prev_y
  dtheta = actual_player_angle - prev_angle

  # Normalize angular difference to [-pi, pi]
  dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

  # Define a reference unit vector (e.g., forward direction)
  ref_vector = np.array([np.cos(prev_angle), np.sin(prev_angle)])
  velocity_vector = np.array([dx, dy])
  v_linear = np.dot(velocity_vector, ref_vector) / dt
  
  v_angular = (dtheta / dt + np.pi) % (2 * np.pi) - np.pi             # Angular velocity (change in angle per time)

  # Create control input
  control_input = [v_linear, v_angular]

  # # Perform prediction
  # predict(ekf, control_input, dt)
  
  # Calculate player's speed (with respect to sensors, noise should be presented)
  x_speed = add_noise(dx / dt, sigma=ENCODER_MEASUREMENT_NOISE[0])
  y_speed = add_noise(dy / dt, sigma=ENCODER_MEASUREMENT_NOISE[1])
  # Calculate angular speed with added noise and normalize the angle
  theta_speed = add_noise(v_angular, sigma=IMU_NOISE[2])
  theta_speed = (theta_speed + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]

  # Pack measurements together
  measurements = [
    x_speed,
    y_speed,
    theta_speed,
    *np.ravel(noisy_lidar_detections)
  ]
  
  # SLAM Update Step: Refine player's position using noisy measurements
  update(ekf, control_input, measurements, dt, DETECTION_THRESHOLD)

  print(ekf.get_state())
  # Extract estimated player position from EKF state
  estimated_player_x, estimated_player_y, estimated_player_angle = ekf.get_state().flatten()[:3]

  # Render frame
  SCREEN.fill(WHITE)

  # Draw actual player
  draw_player(actual_player_x, actual_player_y, actual_player_angle)
  
  # Draw obstacles and uncertainty circles
  draw_obstacles(OBSTACLES)
  draw_estimated_positions(ekf.get_state(), ekf.get_covariance())

  # Draw LiDAR detections
  draw_lidar_detections(estimated_player_x, estimated_player_y, estimated_player_angle, noisy_lidar_detections)

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
