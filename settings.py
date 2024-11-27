import pygame

# Screen settings
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Player settings
PLAYER_SIZE = 40
PLAYER_SPEED = 5  # Units per frame
ROTATION_SPEED = 5  # Degrees per frame
PLAYER_PROCESS_NOISE = [0.01, 0.01, 0.001]  # [x_noise, y_noise, angle_noise]

# LiDAR settings
LIDAR_RANGE = 200  # Detection range in units
LIDAR_FOV = 90  # Field of view in degrees
LIDAR_PROCESS_NOISE = [0, 0]  # [distance_noise, angle_noise]
LIDAR_MEASUREMENT_NOISE = [0.1, 0.01]  # [distance_noise, angle_noise]

# Wheel encoder settings
ENCODER_MEASUREMENT_NOISE = [2, 2] # [x_speed, y_speed]

# Detection Threshold
DETECTION_THRESHOLD = 150

# IMU
IMU_NOISE = [0, 0, 0.45]

# Obstacles
# OBSTACLES = [(300, 300), (400, 500), (700, 100), (500, 300)]
OBSTACLES = [(664, 397), (579, 546), (236, 37), (204, 393), (524, 112), (544, -114), (166, 747), (634, -32), (356, 459), (604, 361)]

# EKF settings
# EKF_PROCESS_NOISE = [0.0] * EKF_STATE_DIM  # Process noise for all states
# EKF_MEASUREMENT_NOISE = [1.0] * EKF_MEAS_DIM  # Measurement noise for all measurements

# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Player Movement with SLAM")
