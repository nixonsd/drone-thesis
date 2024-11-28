import numpy as np
import pygame
import math
from settings import SCREEN, PLAYER_SIZE, WHITE, BLUE, RED, GREEN

def draw_text(screen, text, position, font_size=24, color=(0, 0, 0)):
  """
  Draw text on the Pygame screen.
  :param screen: Pygame screen surface.
  :param text: Text to display.
  :param position: Tuple (x, y) for the text position.
  :param font_size: Size of the font.
  :param color: Color of the text (default is black).
  """
  font = pygame.font.Font(None, font_size)
  text_surface = font.render(text, True, color)
  screen.blit(text_surface, position)

# def draw_uncertainty_circles(state, covariance, scale=10):
#   """
#   Draw circles around the robot and obstacles to represent uncertainty.
#   :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
#   :param covariance: EKF covariance matrix.
#   :param scale: Factor to scale the circle's radius for better visibility.
#   """
#   # Draw circle for the robot position (index 0 and 1)
#   robot_x, robot_y = state[0, 0], state[1, 0]
#   robot_cov = covariance[0:2, 0:2]  # Covariance for robot position
#   robot_std_dev = np.sqrt(np.trace(robot_cov))
#   robot_radius = 100 - int(robot_std_dev * scale)
#   pygame.draw.circle(SCREEN, GREEN, (int(robot_x), int(robot_y)), robot_radius, 1)

#   # Loop over obstacles (start from index 3)
#   for i in range(3, state.shape[0], 2):
#     x, y = state[i, 0], state[i + 1, 0]

#     # Extract 2x2 covariance block for this obstacle
#     P_block = covariance[i:i + 2, i:i + 2]

#     # Calculate the standard deviation (radius of the uncertainty circle)
#     std_dev = np.sqrt(np.trace(P_block))

#     # Scale the radius for better visibility
#     radius = 50 - int(std_dev * scale)

#     # Draw the circle
#     pygame.draw.circle(SCREEN, GREEN, (int(x), int(y)), radius, 1)

import pygame
import numpy as np

# def draw_uncertainty_ellipses(state, covariance, std_devs=3, scale=1):
#     """
#     Draw ellipses around the robot and obstacles to represent uncertainty.
#     :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
#     :param covariance: EKF covariance matrix.
#     :param std_devs: Number of standard deviations for ellipse size.
#     :param scale: Factor to scale the ellipse dimensions for better visibility.
#     """
#     def draw_ellipse(x, y, cov, color):
#         # Calculate eigenvalues and eigenvectors of the covariance matrix
#         eigenvalues, eigenvectors = np.linalg.eig(cov)
#         angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

#         # Scale eigenvalues for desired confidence interval
#         width, height = 2 * std_devs * np.sqrt(eigenvalues) * scale

#         # Draw the ellipse
#         ellipse_rect = pygame.Rect(0, 0, width, height)
#         ellipse_rect.center = (x, y)
#         pygame.draw.ellipse(SCREEN, color, ellipse_rect, 1)
#         pygame.draw.line(SCREEN, color, (x, y), 
#                          (x + width / 2 * np.cos(np.radians(angle)),
#                           y + width / 2 * np.sin(np.radians(angle))), 1)

#     # Draw ellipse for the robot position (index 0 and 1)
#     robot_x, robot_y = state[0, 0], state[1, 0]
#     robot_cov = covariance[0:2, 0:2]  # Covariance for robot position
#     draw_ellipse(robot_x, robot_y, robot_cov, GREEN)

#     # Loop over obstacles (start from index 3)
#     for i in range(3, state.shape[0], 2):
#         x, y = state[i, 0], state[i + 1, 0]

#         # Extract 2x2 covariance block for this obstacle
#         P_block = covariance[i:i + 2, i:i + 2]

#         # Draw the ellipse for this obstacle
#         draw_ellipse(x, y, P_block, GREEN)

# def draw_uncertainty_ellipses(state, covariance, std_devs=3, scale=1, color=GREEN):
#     """
#     Draw ellipses around the robot and obstacles to represent uncertainty.
#     :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
#     :param covariance: EKF covariance matrix.
#     :param std_devs: Number of standard deviations for ellipse size.
#     :param scale: Factor to scale the ellipse dimensions for better visibility.
#     :param color: Color for the ellipse and line.
#     """
#     def draw_ellipse(x, y, cov, color):
#         # Stabilize covariance matrix
#         cov = (cov + cov.T) / 2

#         # Calculate eigenvalues and eigenvectors of the covariance matrix
#         eigenvalues, eigenvectors = np.linalg.eig(cov)
#         if np.any(eigenvalues < 0):
#             print("Warning: Covariance matrix has negative eigenvalues")
#             return

#         # Determine angle of rotation
#         angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

#         # Scale eigenvalues for desired confidence interval
#         width, height = 2 * std_devs * np.sqrt(eigenvalues) * scale

#         # Draw the ellipse
#         ellipse_rect = pygame.Rect(0, 0, int(width), int(height))
#         ellipse_rect.center = (int(x), int(y))
#         pygame.draw.ellipse(SCREEN, color, ellipse_rect, 1)

#         # Draw orientation line (major axis)
#         pygame.draw.line(SCREEN, color, (int(x), int(y)), 
#                          (int(x + width / 2 * np.cos(np.radians(angle))),
#                           int(y + width / 2 * np.sin(np.radians(angle)))), 1)

#     # Draw ellipse for the robot position (index 0 and 1)
#     robot_x, robot_y = state[0, 0], state[1, 0]
#     robot_cov = covariance[0:2, 0:2]  # Covariance for robot position
#     draw_ellipse(robot_x, robot_y, robot_cov, color)

#     # Loop over obstacles (start from index 3)
#     for i in range(3, state.shape[0], 2):
#         x, y = state[i, 0], state[i + 1, 0]

#         # Extract 2x2 covariance block for this obstacle
#         P_block = covariance[i:i + 2, i:i + 2]

#         # Draw the ellipse for this obstacle
#         draw_ellipse(x, y, P_block, color)

import numpy as np
import pygame

def draw_uncertainty_ellipses(state, covariance, std_devs=3, scale=1, color=(0, 255, 0)):
    """
    Draw ellipses around the robot and obstacles to represent uncertainty.
    :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
    :param covariance: EKF covariance matrix.
    :param std_devs: Number of standard deviations for ellipse size.
    :param scale: Factor to scale the ellipse dimensions for better visibility.
    :param color: Color for the ellipse and line.
    """

    def draw_ellipse(x, y, cov, color):
        # Ensure covariance is symmetric
        cov = (cov + cov.T) / 2

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        if np.any(eigenvalues < 0):
            print("Warning: Covariance matrix has negative eigenvalues")
            return

        # Eigenvalues give the axes' lengths, scaled by desired standard deviations
        width, height = 2 * std_devs * np.sqrt(eigenvalues) * scale

        # Eigenvectors give the orientation
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Create a surface for rotating the ellipse
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.ellipse(surface, color, (0, 0, width, height), 1)

        # Rotate the surface
        rotated_surface = pygame.transform.rotate(surface, angle)

        # Draw the rotated ellipse on the screen
        SCREEN.blit(rotated_surface, rotated_surface.get_rect(center=(int(x), int(y))))

    # Draw ellipse for the robot position (index 0 and 1)
    robot_x, robot_y = state[0, 0], state[1, 0]
    robot_cov = covariance[0:2, 0:2]  # Covariance for robot position
    draw_ellipse(robot_x, robot_y, robot_cov, color)

    # Loop over obstacles (start from index 3)
    for i in range(3, state.shape[0], 2):
        x, y = state[i, 0], state[i + 1, 0]

        # Extract 2x2 covariance block for this obstacle
        P_block = covariance[i:i + 2, i:i + 2]

        # Draw the ellipse for this obstacle
        draw_ellipse(x, y, P_block, color)

def draw_estimated_positions(state, covariance, color=(0, 255, 0), size=10, thickness=1):
    """
    Draw crosshairs (crossover markers) at the estimated positions.
    :param state: EKF state vector [robot_x, robot_y, robot_angle, obstacle_1_x, obstacle_1_y, ...].
    :param covariance: EKF covariance matrix (unused, but kept for consistency with the ellipse version).
    :param color: Color for the crosshairs.
    :param size: Size of each crosshair (length of the lines).
    :param thickness: Thickness of the crosshair lines.
    """
    def draw_cross(x, y, color, size, thickness):
        # Draw horizontal and vertical lines to create the crosshair
        pygame.draw.line(SCREEN, color, (int(x - size / 2), int(y)), (int(x + size / 2), int(y)), thickness)
        pygame.draw.line(SCREEN, color, (int(x), int(y - size / 2)), (int(x), int(y + size / 2)), thickness)

    # Draw cross for the robot position (index 0 and 1)
    robot_x, robot_y = state[0, 0], state[1, 0]
    draw_cross(robot_x, robot_y, color, size, thickness)

    # Loop over obstacles (start from index 3)
    for i in range(3, state.shape[0], 2):
        x, y = state[i, 0], state[i + 1, 0]
        draw_cross(x, y, color, size, thickness)


# def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
#     """
#     Create a plot of the covariance confidence ellipse of *x* and *y*.

#     Parameters
#     ----------
#     x, y : array-like, shape (n, )
#         Input data.

#     ax : matplotlib.axes.Axes
#         The Axes object to draw the ellipse into.

#     n_std : float
#         The number of standard deviations to determine the ellipse's radiuses.

#     **kwargs
#         Forwarded to `~matplotlib.patches.Ellipse`

#     Returns
#     -------
#     matplotlib.patches.Ellipse
#     """
#     if x.size != y.size:
#         raise ValueError("x and y must be the same size")

#     cov = np.cov(x, y)
#     pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#     # Using a special case to obtain the eigenvalues of this
#     # two-dimensional dataset.
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
#                       facecolor=facecolor, **kwargs)

#     # Calculating the standard deviation of x from
#     # the squareroot of the variance and multiplying
#     # with the given number of standard deviations.
#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     mean_x = np.mean(x)

#     # calculating the standard deviation of y ...
#     scale_y = np.sqrt(cov[1, 1]) * n_std
#     mean_y = np.mean(y)

#     transf = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean_x, mean_y)

#     ellipse.set_transform(transf + ax.transData)
#     return ax.add_patch(ellipse)

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
