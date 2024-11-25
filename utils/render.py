import pygame
import math
from settings import SCREEN, PLAYER_SIZE, WHITE, BLUE, RED, GREEN

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
