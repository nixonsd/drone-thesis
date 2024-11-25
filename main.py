import pygame
import numpy as np
from settings import *
from utils.render import draw_player, draw_obstacles, draw_lidar_detections, draw_uncertainty_circles
from models.lidar_simulator import simulate_lidar
from models.ekf import ExtendedKalmanFilter
from utils.slam import measurement_model, motion_model, predict, update

# EKF setup
ekf = ExtendedKalmanFilter(state_dim=3, meas_dim=3)
ekf.set_process_noise(np.diag([0.1] * 3))
ekf.set_measurement_noise(np.diag([1.0] * 3))

# player_x, player_y, player_angle = 0, 0, 0

# Player actual position (ground truth)
actual_player_x, actual_player_y, actual_player_angle = 50, 50, 0

# Player estimated position (SLAM output)
estimated_player_x, estimated_player_y, estimated_player_angle = 0, 0, 0

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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

    actual_player_angle = (actual_player_angle + np.pi) % (2 * np.pi) - np.pi

    # Simulate LiDAR detections (ground truth position is used here)
    lidar_detections = simulate_lidar(
      actual_player_x, actual_player_y, actual_player_angle, OBSTACLES, LIDAR_RANGE, LIDAR_FOV
    )
    
    # SLAM Prediction Step
    predict(ekf, motion_model, [0, 0, 0])

    # SLAM Update Step
    correspondences, unmatched = update(
      ekf, actual_player_x, actual_player_y, actual_player_angle, lidar_detections, measurement_model
    )

    # Debugging info
    # print("Correspondences:", len(correspondences), correspondences)
    if len(unmatched):
      print("Unmatched:", len(unmatched), unmatched)

    # Render frame
    SCREEN.fill(WHITE)

    # Draw player and obstacles
    draw_player(actual_player_x, actual_player_y, actual_player_angle)
    draw_obstacles(OBSTACLES)

    # Draw uncertainty circles for robot and obstacles (excluding robot angle)
    draw_uncertainty_circles(ekf.get_state(), ekf.get_covariance())

    # Draw LiDAR detections
    draw_lidar_detections(actual_player_x, actual_player_y, actual_player_angle, lidar_detections)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
