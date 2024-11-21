# import matplotlib.pyplot as plt
# import numpy as np

# # standard deviation (+-3m)
# SIGMA = 5

# # with this function you can easily change dimension of data
# # cov is [[var(x), cov(x, y)], [cov(x, y), var(y)]]
# # var(x) is sigma ** 2
# def uncertainity_add(distance, sigma):
#     mean = np.array([distance])
#     covariance = np.diag([sigma ** 2])
#     distance = np.random.multivariate_normal(mean, covariance)[0]  # Extract the scalar value
#     distance = max(distance, 0)  # Ensure non-negative
#     return distance

# def linear_function(x):
#     return 3 * x

# def main():
#     x = np.arange(0, 100, 5)
#     y_real = linear_function(x)  # Real data
#     y_uncertain = [uncertainity_add(yi, SIGMA) for yi in y_real]
    
#     # Plot real data as dashed line
#     plt.plot(x, y_real, '--', label='Real data')
    
#     # Plot uncertain data
#     plt.plot(x, y_uncertain, label='Uncertain data')
    
#     plt.xlabel('t')
#     plt.ylabel('Distance')
#     plt.legend(loc="upper left")
#     plt.title('Read GPS signals')
#     plt.show()

# if __name__ == '__main__':
#     main()


import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
  def __init__(self, F, B, H, Q, R, x0, P0):
    self.F = F  # State transition model
    self.B = B  # Control input model
    self.H = H  # Observation model
    self.Q = Q  # Process noise covariance
    self.R = R  # Measurement noise covariance
    self.x = x0  # Initial state estimate
    self.P = P0  # Initial covariance estimate

  def predict(self, u):
    # Predict the state and state covariance
    self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
    self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    return self.x

  def update(self, z):
    # Compute the Kalman gain
    S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
    K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    # Update the state estimate and covariance matrix
    y = z - np.dot(self.H, self.x)  # Measurement residual
    self.x = self.x + np.dot(K, y)
    I = np.eye(self.P.shape[0])
    self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    return self.x


def uncertainty_add(distance, sigma):
  """
  Add Gaussian noise to the data to simulate GPS noise.
  """
  mean = np.array([distance])
  covariance = np.diag([sigma ** 2])
  distance = np.random.multivariate_normal(mean, covariance)[0]  # Extract scalar value
  distance = max(distance, 0)  # Ensure non-negative
  return distance


def linear_function(x):
  """
  Ground truth function: Distance = 3 * time.
  """
  acceleration = 1/3
  velocity = acceleration * x
  distance = 0.5 * acceleration * x**2
  
  return distance
  
  # return 3 * x
  """
  Ground truth function: Distance = 3 * time^2 / 2, Velocity = 3 * time, Acceleration = 3.
  """
  # acceleration = 1
  # velocity = acceleration * x
  # distance = 0.5 * acceleration * x**2
  # return distance, velocity, acceleration * np.ones_like(x)  # Distance, Velocity, Acceleration

def calculate_mse(real, predicted):
  """
  Calculate Mean Squared Error (MSE).
  """
  return np.mean((np.array(real) - np.array(predicted)) ** 2)

def main():
  # Time steps and true positions
  x = np.arange(0, 20, 0.5)
  y_real = linear_function(x)  # Ground truth
  y_noisy = np.array([uncertainty_add(y, 3) for y in y_real])  # Noisy GPS measurements

  # Kalman filter parameters
  dt = 0.5  # Time step size
  F = np.array([[1, dt], [0, 1]])  # State transition model
  B = np.array([[0], [0]])  # No control input
  H = np.array([[1, 0]])  # Observation model
  Q = np.array([[1, 0], [0, 1]])  # Process noise covariance
  R = np.array([[25]])  # Measurement noise covariance
  x0 = np.array([[0], [0]])  # Initial state (position = 0, velocity = 0)
  P0 = np.array([[1, 0], [0, 1]])  # Initial state covariance

  # Create Kalman filter instance
  kf = KalmanFilter(F, B, H, Q, R, x0, P0)

  # Apply Kalman filter
  y_filtered = []
  for z in y_noisy:
    kf.predict(np.array([[0]]))  # No control input
    filtered_state = kf.update(np.array([[z]]))
    y_filtered.append(filtered_state[0, 0])  # Save filtered position

  # Calculate MSE
  mse_no_filter = calculate_mse(y_real, y_noisy)
  mse_with_filter = calculate_mse(y_real, y_filtered)

  # Create subplots
  fig = plt.figure(figsize=(12, 8))

  # First row: Full-width plot
  ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
  ax1.plot(x, y_real, '--', label='Real data')
  ax1.plot(x, y_noisy, label='Noisy measurements')
  ax1.plot(x, y_filtered, label='Filtered data (Kalman)', linewidth=2)
  ax1.set_title('GPS Signal Filtering with Kalman Filter')
  ax1.set_xlabel('Time (t)')
  ax1.set_ylabel('Distance')
  ax1.legend(loc="upper left")

  # Second row, left: Difference noisy vs real
  ax2 = plt.subplot2grid((2, 2), (1, 0))
  ax2.plot(x, np.array(y_noisy) - np.array(y_real), label=f'Noisy - Real (MSE={mse_no_filter:.2f})', color='orange')
  ax2.set_title('Difference: Noisy Measurements vs Real Data')
  ax2.set_xlabel('Time (t)')
  ax2.set_ylabel('Difference')
  ax2.legend(loc="upper left")

  # Second row, right: Difference filtered vs real
  ax3 = plt.subplot2grid((2, 2), (1, 1))
  ax3.plot(x, np.array(y_filtered) - np.array(y_real), label=f'Filtered - Real (MSE={mse_with_filter:.2f})', color='green')
  ax3.set_title('Difference: Filtered Data vs Real Data')
  ax3.set_xlabel('Time (t)')
  ax3.set_ylabel('Difference')
  ax3.legend(loc="upper left")

  # Adjust layout
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
