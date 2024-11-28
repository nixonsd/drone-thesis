import pygame

# Screen settings
WIDTH, HEIGHT = 800, 600  # Simulates a 20x15m area (scale: 1 unit = 0.025 meters)
FPS = 60  # 60 frames per second, typical for smooth simulations

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Obstacles
OBSTACLES = [
    (150, 150), (300, 300), (600, 450), (700, 200)
]  # Distributed across the map

# Player settings
PLAYER_SIZE = 40  # Approximately 1 meter in diameter (scaled)
PLAYER_SPEED = 3  # 3 meters per second (scaled units per frame)
ROTATION_SPEED = 2  # 2 degrees per frame, equivalent to 120 degrees/second
PLAYER_PROCESS_NOISE = [0.05, 0.05, 0.0005]  # Reduced noise for more precise control

# LiDAR settings
LIDAR_RANGE = 250  # Detection range of 500 units (~12.5 meters)
LIDAR_FOV = 270  # Field of view increased to match common LiDARs
LIDAR_PROCESS_NOISE = [0.005, 0.0001]  # Reduced process noise
LIDAR_MEASUREMENT_NOISE = [0.05, 0.005]  # More realistic noise for distance and angle

# Wheel encoder settings
ENCODER_MEASUREMENT_NOISE = [0.1, 0.1]  # Smaller noise for high-resolution encoders

# Detection Threshold
DETECTION_THRESHOLD = 300  # Higher threshold for obstacle proximity

# IMU settings
IMU_NOISE = [0.005, 0.005, 0.05]  # Reduced noise for modern IMUs


# EKF settings (if using an Extended Kalman Filter)
EKF_PROCESS_NOISE = [0.01, 0.01, 0.0001]  # Lower values for x, y, and angle noise

EKF_MEASUREMENT_NOISE = [0.05, 0.05, 0.001]  # Lower measurement noise for x, y, and angle

# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Player Movement with SLAM")
