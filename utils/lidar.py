import math

def get_object_position(robot_x, robot_y, robot_angle, distance, angle):
  absolute_angle = angle + robot_angle
  x_global = robot_x + distance * math.cos(absolute_angle)
  y_global = robot_y + distance * math.sin(absolute_angle)
  return x_global, y_global

def process_lidar_detections(robot_x, robot_y, robot_angle, lidar_detections):
  obstacle_coordinates = []
  for distance, angle in lidar_detections:
    x_global, y_global = get_object_position(robot_x, robot_y, robot_angle, distance, angle)
    obstacle_coordinates.extend([x_global, y_global])
  return obstacle_coordinates
