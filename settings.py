import pygame

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Clock settings
FPS = 60

# Player settings
PLAYER_SIZE = 40
PLAYER_SPEED = 5
ROTATION_SPEED = 5

# LiDAR settings
LIDAR_RANGE = 200
LIDAR_FOV = 90  # Degrees

# Obstacles
OBSTACLES = [(10, 10), (300, 300), (400, 500), (700, 100), (500, 300)]

# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Player Movement with SLAM")
