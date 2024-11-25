import pygame
import numpy as np
from settings import *
from utils.render import draw_player, draw_obstacles, draw_lidar_detections
from models.lidar_simulator import simulate_lidar
from models.ekf import ExtendedKalmanFilter
from utils.slam import measurement_model, motion_model, predict, update

# EKF setup
ekf = ExtendedKalmanFilter(state_dim=3, meas_dim=3)
ekf.set_process_noise(np.diag([0.5] * 3))
ekf.set_measurement_noise(np.diag([1.0] * 3))

player_x, player_y, player_angle = 0, 0, 0
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_x += PLAYER_SPEED * np.cos(player_angle)
        player_y += PLAYER_SPEED * np.sin(player_angle)
    if keys[pygame.K_DOWN]:
        player_x -= PLAYER_SPEED * np.cos(player_angle)
        player_y -= PLAYER_SPEED * np.sin(player_angle)
    if keys[pygame.K_LEFT]:
        player_angle -= np.radians(ROTATION_SPEED)
    if keys[pygame.K_RIGHT]:
        player_angle += np.radians(ROTATION_SPEED)

    player_angle = (player_angle + np.pi) % (2 * np.pi) - np.pi

    # Simulate LiDAR detections
    lidar_detections = simulate_lidar(player_x, player_y, player_angle, OBSTACLES, LIDAR_RANGE, LIDAR_FOV)

    # SLAM Prediction Step
    predict(ekf, motion_model)

    # SLAM Update Step
    correspondences, unmatched = update(
      ekf, player_x, player_y, player_angle, lidar_detections, measurement_model
    )

    # Debugging info
    print("Correspondences:", len(correspondences), correspondences)
    print("Unmatched:", len(unmatched), unmatched)

    SCREEN.fill(WHITE)
    draw_player(player_x, player_y, player_angle)
    draw_obstacles(OBSTACLES)
    draw_lidar_detections(player_x, player_y, player_angle, lidar_detections)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
