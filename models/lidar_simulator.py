# import pygame
# import sys

# # Initialize pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Circle Game")

# # Colors
# WHITE = (255, 255, 255)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)

# # Game clock
# clock = pygame.time.Clock()
# FPS = 60

# # Player circle properties
# player_pos = [WIDTH // 2, HEIGHT // 2]
# player_radius = 20
# player_speed = 5

# # Hardcoded static circles
# static_circles = [
#     (200, 150, 30),  # (x, y, radius)
#     (600, 100, 30),
#     (300, 400, 30),
#     (700, 500, 30),
#     (400, 300, 30),
# ]

# # Main game loop
# def main():
#     global player_pos
#     running = True

#     while running:
#         screen.fill(WHITE)

#         # Event handling
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         # Movement
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_w]:  # Move up
#             player_pos[1] -= player_speed
#         if keys[pygame.K_s]:  # Move down
#             player_pos[1] += player_speed
#         if keys[pygame.K_a]:  # Move left
#             player_pos[0] -= player_speed
#         if keys[pygame.K_d]:  # Move right
#             player_pos[0] += player_speed

#         # Boundary check
#         player_pos[0] = max(player_radius, min(WIDTH - player_radius, player_pos[0]))
#         player_pos[1] = max(player_radius, min(HEIGHT - player_radius, player_pos[1]))

#         # Draw player circle
#         pygame.draw.circle(screen, BLUE, player_pos, player_radius)

#         # Draw static circles
#         for circle in static_circles:
#             pygame.draw.circle(screen, RED, (circle[0], circle[1]), circle[2])

#         # Update display
#         pygame.display.flip()

#         # Cap the frame rate
#         clock.tick(FPS)

#     pygame.quit()
#     sys.exit()

# if __name__ == "__main__":
#     main()


# import math

# def simulate_lidar_with_occlusion(sensor_position, points, fov, max_range):
#     """
#     Simulate a 2D lidar sensor with occlusion detection.
    
#     :param sensor_position: Tuple (x, y, theta) representing the lidar's position and orientation
#     :param points: List of tuples [(x, y), ...] representing the coordinates of points in the plane
#     :param fov: Field of view in degrees (e.g., 90 for 90 degrees)
#     :param max_range: Maximum range of the lidar sensor
#     :return: List of detected points as (distance, angle)
#     """
#     sensor_x, sensor_y, sensor_theta = sensor_position
#     detected_points = []

#     # Calculate distance and angle to all points
#     point_data = []
#     for point in points:
#         point_x, point_y = point
#         dx = point_x - sensor_x
#         dy = point_y - sensor_y
#         distance = math.sqrt(dx**2 + dy**2)
#         angle = math.degrees(math.atan2(dy, dx)) - sensor_theta
#         angle = (angle + 180) % 360 - 180  # Normalize angle to (-180, 180]
        
#         if distance <= max_range and abs(angle) <= fov / 2:
#             point_data.append((distance, angle, point_x, point_y))
    
#     # Sort points by distance (nearest first)
#     point_data.sort()

#     # Check for occlusion and retain only visible points
#     detected_angles = set()
#     for distance, angle, point_x, point_y in point_data:
#         # If this angle hasn't been covered by a closer point, add it
#         if angle not in detected_angles:
#             detected_points.append((distance, angle))
#             detected_angles.add(angle)

#     return detected_points


# # Define the LiDAR sensor's position and parameters
# sensor_position = (0, 0, 0)  # (x, y, theta) where theta is orientation in degrees
# fov = 90  # Field of view in degrees
# max_range = 10  # Maximum range of the sensor

# # Define points on the XY plane
# points = [
#     (5, 5),    # In range and visible
#     (8, 0),    # In range and visible
#     (5, 5),    # Same point, should only be detected once
#     (3, -4),   # In range and visible
#     (2, -2),   # Closer than (3, -4), will occlude it
#     (-2, 3),   # Outside the FOV, won't be detected
#     (9, 9),    # Out of range
# ]

# # Simulate the LiDAR sensor
# detected_points = simulate_lidar_with_occlusion(sensor_position, points, fov, max_range)

# # Print detected points
# print("Detected Points:")
# for distance, angle in detected_points:
#     print(f"Distance: {distance:.2f}, Angle: {angle:.2f} degrees")


# import pygame
# import math

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Player Square Movement with LiDAR Sensor")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)

# # Clock for controlling the frame rate
# clock = pygame.time.Clock()
# FPS = 60

# # Player settings
# player_size = 40
# player_x, player_y = WIDTH // 2, HEIGHT // 2  # Initial position
# player_angle = 0  # Initial angle
# player_speed = 5  # Movement speed
# rotation_speed = 5  # Rotation speed

# # LiDAR settings
# lidar_range = 200  # LiDAR detection range
# lidar_fov = 90  # Field of view in degrees
# obstacles = [(300, 300), (400, 500), (700, 100), (500, 300)]  # Predefined obstacles

# # Game loop
# running = True

# while running:
#     # Event handling
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Key states
#     keys = pygame.key.get_pressed()

#     # Control array [x, y, angle]
#     control_array = [player_x, player_y, player_angle]

#     # Movement and rotation logic
#     if keys[pygame.K_UP]:
#         player_x += player_speed * math.cos(math.radians(player_angle))
#         player_y += player_speed * math.sin(math.radians(player_angle))
#     if keys[pygame.K_DOWN]:
#         player_x -= player_speed * math.cos(math.radians(player_angle))
#         player_y -= player_speed * math.sin(math.radians(player_angle))
#     if keys[pygame.K_LEFT]:
#         player_angle -= rotation_speed
#     if keys[pygame.K_RIGHT]:
#         player_angle += rotation_speed

#     # Normalize angle to keep it between 0 and 360
#     player_angle %= 360
#     control_array = [round(player_x, 2), round(player_y, 2), round(player_angle, 2)]

#     # Clear screen
#     screen.fill(WHITE)

#     # Draw player
#     half_size = player_size // 2
#     angle_rad = math.radians(player_angle)
#     corners = [
#         (
#             player_x + half_size * math.cos(angle_rad) - half_size * math.sin(angle_rad),
#             player_y + half_size * math.sin(angle_rad) + half_size * math.cos(angle_rad),
#         ),
#         (
#             player_x - half_size * math.cos(angle_rad) - half_size * math.sin(angle_rad),
#             player_y - half_size * math.sin(angle_rad) + half_size * math.cos(angle_rad),
#         ),
#         (
#             player_x - half_size * math.cos(angle_rad) + half_size * math.sin(angle_rad),
#             player_y - half_size * math.sin(angle_rad) - half_size * math.cos(angle_rad),
#         ),
#         (
#             player_x + half_size * math.cos(angle_rad) + half_size * math.sin(angle_rad),
#             player_y + half_size * math.sin(angle_rad) - half_size * math.cos(angle_rad),
#         ),
#     ]
#     pygame.draw.polygon(screen, BLUE, corners)

#     # Draw obstacles
#     for obstacle in obstacles:
#         pygame.draw.circle(screen, RED, obstacle, 10)

#     # Simulate LiDAR
#     lidar_detections = []
#     for obstacle in obstacles:
#         dx = obstacle[0] - player_x
#         dy = obstacle[1] - player_y
#         distance = math.sqrt(dx**2 + dy**2)
#         angle_to_obstacle = math.degrees(math.atan2(dy, dx)) - player_angle
#         angle_to_obstacle = (angle_to_obstacle + 180) % 360 - 180  # Normalize to [-180, 180]

#         if distance <= lidar_range and abs(angle_to_obstacle) <= lidar_fov / 2:
#             lidar_detections.append((distance, angle_to_obstacle))

#     # Draw LiDAR detections
#     for detection in lidar_detections:
#         distance, angle = detection
#         end_x = player_x + distance * math.cos(math.radians(player_angle + angle))
#         end_y = player_y + distance * math.sin(math.radians(player_angle + angle))
#         pygame.draw.line(screen, GREEN, (player_x, player_y), (end_x, end_y), 2)

#     # Display the control array and lidar data
#     font = pygame.font.Font(None, 36)
#     control_text = font.render(f"Control Array: {control_array}", True, BLACK)
#     screen.blit(control_text, (10, 10))

#     lidar_text = font.render(f"LiDAR: {len(lidar_detections)} detections", True, BLACK)
#     screen.blit(lidar_text, (10, 50))

#     # Update the display
#     pygame.display.flip()

#     # Cap the frame rate
#     clock.tick(FPS)

# pygame.quit()


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

