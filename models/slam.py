import numpy as np
from sympy import symbols, Matrix
from .ekf import ExtendedKalmanFilter
class SLAM:
  def __init__(self, player_pos, obstacles, process_noise, measurement_noise):
    """
    Initialize the SLAM module.
    :param player_pos: Initial player position [x, y, theta]
    :param obstacles: Initial positions of obstacles [[x1, y1], [x2, y2], ...]
    :param process_noise: Process noise covariance matrix
    :param measurement_noise: Measurement noise covariance matrix
    """
    self.state_dim = 3 + 2 * len(obstacles)  # Player state + obstacle positions
    self.meas_dim = 2 * len(obstacles)      # LiDAR provides distance and angle to each obstacle
    self.obstacle_count = len(obstacles)

    # Initialize EKF with the state dimension and measurement dimension
    self.ekf = ExtendedKalmanFilter(self.state_dim, self.meas_dim, initial_state=self._initialize_state(player_pos, obstacles))

    # Set noise matrices
    self.ekf.set_process_noise(process_noise)
    self.ekf.set_measurement_noise(measurement_noise)

  def _initialize_state(self, player_pos, obstacles):
    """
    Initialize the state vector.
    :param player_pos: Player position [x, y, theta]
    :param obstacles: Obstacle positions [[x1, y1], [x2, y2], ...]
    :return: State vector
    """
    state = np.zeros((self.state_dim, 1))
    state[0:3, 0] = player_pos
    for i, obstacle in enumerate(obstacles):
      state[3 + 2 * i:3 + 2 * i + 2, 0] = obstacle
    return state

  def predict(self, control):
    """
    Perform the prediction step.
    :param control: Control input [dx, dy, dtheta]
    """
    def motion_model(state, u=None):
      """
      State transition model for the player and obstacles.
      :param state: Current state vector (numeric or symbolic)
      :param u: Control input [dx, dy, dtheta] (numeric or symbolic)
      :return: Updated state vector
      """
      from sympy import cos, sin, symbols, Matrix

      if isinstance(state, np.ndarray):  # Numeric case
          x, y, theta = state[0, 0], state[1, 0], state[2, 0]
          dx, dy, dtheta = u
          x += dx * np.cos(theta) - dy * np.sin(theta)
          y += dx * np.sin(theta) + dy * np.cos(theta)
          theta += dtheta
          state[0, 0], state[1, 0], state[2, 0] = x, y, theta
          return state
      else:  # Symbolic case (SymPy)
          dx, dy, dtheta = symbols('dx dy dtheta')  # Symbolic control inputs
          x, y, theta = state[0], state[1], state[2]
          x += dx * cos(theta) - dy * sin(theta)
          y += dx * sin(theta) + dy * cos(theta)
          theta += dtheta

          # Replace rows explicitly to avoid dimension mismatch
          state[0] = x
          state[1] = y
          state[2] = theta
          return state

    self.ekf.predict(motion_model, u=control)

  def update(self, lidar_measurements):
    """
    Perform the update step.
    :param lidar_measurements: LiDAR readings [(distance1, angle1), (distance2, angle2), ...]
    """
    def measurement_model(state):
      """
      Measurement model mapping state to LiDAR observations.
      :param state: Current state vector
      :return: Predicted measurements
      """
      x, y, theta = state[0, 0], state[1, 0], state[2, 0]
      measurements = []
      for i in range(self.obstacle_count):
        obs_x, obs_y = state[3 + 2 * i, 0], state[3 + 2 * i + 1, 0]
        dx, dy = obs_x - x, obs_y - y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - theta
        measurements.extend([distance, angle])
      return np.array(measurements).reshape(-1, 1)

    # Flatten LiDAR measurements for the update step
    z = np.array(lidar_measurements).flatten().reshape(-1, 1)
    self.ekf.update(z, measurement_model)

  def get_state(self):
    """
    Get the current state estimate.
    :return: State vector
    """
    return self.ekf.get_state()
