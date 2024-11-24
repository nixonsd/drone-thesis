import pygame
import numpy as np
import cv2
from time import time

# Function to map image to binary array
def map_image_to_array(image_path, threshold=128, scale_factor=0.1):
    """
    Converts a room map image to a binary array with reduced size.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load the image. Check the path.")

    # Resize the image
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Binarize the image
    _, binary_image = cv2.threshold(resized_image, threshold, 1, cv2.THRESH_BINARY)

    return binary_image

# Load the map
image_path = './map.jpg'  # Path to the image
binary_map = map_image_to_array(image_path, scale_factor=0.8)

# Constants
PLAYER_RADIUS = 5
PLAYER_SPEED = 100  # Speed in pixels per second
FPS = 60

ROWS, COLS = binary_map.shape
WINDOW_WIDTH = COLS
WINDOW_HEIGHT = ROWS

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("High Performance Smooth Movement")
clock = pygame.time.Clock()

# Initial player position
player_pos = [20, 20]  # Start at (20, 20)
player_velocity = [0, 0]  # Velocity for x and y

def draw_map():
    """Draw the binary map."""
    for y in range(ROWS):
        for x in range(COLS):
            color = (255, 255, 255) if binary_map[y][x] == 1 else (0, 0, 0)
            screen.set_at((x, y), color)

def draw_player():
    """Draw the player."""
    pygame.draw.circle(screen, (0, 0, 255), (int(player_pos[0]), int(player_pos[1])), PLAYER_RADIUS)

def can_move_to(x, y):
    """Check if the player can move to the given position."""
    left = int(x - PLAYER_RADIUS)
    right = int(x + PLAYER_RADIUS)
    top = int(y - PLAYER_RADIUS)
    bottom = int(y + PLAYER_RADIUS)

    for px in range(left, right + 1):
        for py in range(top, bottom + 1):
            if 0 <= px < COLS and 0 <= py < ROWS:
                if binary_map[py][px] == 0:  # Obstacle
                    return False
    return True

def update_player(delta_time):
    """Update the player's position based on velocity and delta time."""
    new_x = player_pos[0] + player_velocity[0] * delta_time
    new_y = player_pos[1] + player_velocity[1] * delta_time

    if can_move_to(new_x, player_pos[1]):
        player_pos[0] = new_x
    if can_move_to(player_pos[0], new_y):
        player_pos[1] = new_y

# Main game loop
def main():
    global player_velocity
    last_time = time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Get pressed keys
        keys = pygame.key.get_pressed()
        player_velocity = [0, 0]
        if keys[pygame.K_UP]:
            player_velocity[1] = -PLAYER_SPEED
        if keys[pygame.K_DOWN]:
            player_velocity[1] = PLAYER_SPEED
        if keys[pygame.K_LEFT]:
            player_velocity[0] = -PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            player_velocity[0] = PLAYER_SPEED

        # Calculate delta time
        current_time = time()
        delta_time = current_time - last_time
        last_time = current_time

        # Update player position
        update_player(delta_time)

        # Draw everything
        screen.fill((0, 0, 0))  # Clear the screen
        draw_map()
        draw_player()
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

if __name__ == "__main__":
    main()
