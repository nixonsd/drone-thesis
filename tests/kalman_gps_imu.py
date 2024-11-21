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


def add_noise(data, sigma):
    """
    Add Gaussian noise to the data to simulate sensor noise.
    Handles both scalar and iterable data.
    """
    if np.isscalar(data):  # Check if the input is a scalar (float or int)
        noise = np.random.normal(0, sigma)
        return data + noise
    else:  # Handle iterable data (e.g., list or numpy array)
        noise = np.random.normal(0, sigma, size=len(data))
        return data + noise


def linear_function(x):
    """
    Ground truth function: Distance = 1/3 * time^2 / 2, Velocity = 1/3 * time, Acceleration = 1/3.
    """
    acceleration = 1 / 3
    velocity = acceleration * x
    distance = 0.5 * acceleration * x**2
    return distance, velocity, acceleration * np.ones_like(x)  # Distance, Velocity, Acceleration


def calculate_mse(real, predicted):
    """
    Calculate Mean Squared Error (MSE).
    """
    return np.mean((np.array(real) - np.array(predicted)) ** 2)


def main():
    # Time steps and true positions
    x = np.arange(0, 20, 0.5)
    y_real, v_real, a_real = linear_function(x)  # Ground truth: position, velocity, acceleration
    y_noisy = add_noise(y_real, 3)  # GPS position with noise
    a_noisy = add_noise(a_real, 0.01)  # IMU acceleration with noise

    # Integrate noisy acceleration to estimate velocity and position
    dt = 0.5  # Time step
    v_estimated = np.cumsum(a_noisy) * dt  # Velocity by integrating acceleration
    y_estimated_imu = np.cumsum(v_estimated) * dt  # Position by integrating velocity

    # Kalman filter parameters
    F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])  # State transition model
    B = np.array([[0], [0], [0]])  # No control input
    H = np.array([[1, 0, 0]])  # Observation model: we observe position only (GPS)
    Q = np.eye(3) * 0.1  # Process noise covariance
    R = np.array([[25]])  # Measurement noise covariance for GPS
    x0 = np.array([[0], [0], [0]])  # Initial state: position, velocity, acceleration
    P0 = np.eye(3)  # Initial state covariance

    # Create Kalman filter instance
    kf = KalmanFilter(F, B, H, Q, R, x0, P0)

    # Apply Kalman filter
    y_filtered = []
    for z_pos in y_noisy:
        kf.predict(np.array([[0]]))  # No control input
        filtered_state = kf.update(np.array([[z_pos]]))
        y_filtered.append(filtered_state[0, 0])  # Save filtered position

    # Calculate MSE
    mse_position_no_filter = calculate_mse(y_real, y_noisy)
    mse_imu_estimated_position_no_filter = calculate_mse(y_real, y_estimated_imu)
    mse_position_with_filter = calculate_mse(y_real, y_filtered)

    # Create subplots
    fig = plt.figure(figsize=(12, 8))

    # First row: Position data
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(x, y_real, '--', label='Real position')
    ax1.plot(x, y_noisy, label='Noisy GPS position')
    ax1.plot(x, y_estimated_imu, label='IMU-estimated position (double integration)')
    ax1.plot(x, y_filtered, label='Filtered position (Kalman)', linewidth=2)
    ax1.set_title('Position Filtering with Kalman Filter')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Position')
    ax1.legend(loc="upper left")

    # Second row, left: Difference noisy vs real
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax2.plot(x, np.array(y_noisy) - np.array(y_real), label=f'Noisy - Real (MSE={mse_position_no_filter:.2f})', color='orange')
    ax2.set_title('Difference: Noisy Measurements vs Real Data (GPS)')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Difference')
    ax2.legend(loc="upper left")
    
    # Second row, left: Acceleration data
    # ax2 = plt.subplot2grid((2, 3), (1, 0))
    # ax2.plot(x, a_real, '--', label='Real acceleration')
    # ax2.plot(x, a_noisy, label='Noisy IMU acceleration')
    # ax2.set_title('Acceleration Data')
    # ax2.set_xlabel('Time (t)')
    # ax2.set_ylabel('Acceleration')
    # ax2.legend(loc="upper left")
    
    # Second row, center: Difference filtered vs real
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.plot(x, np.array(y_estimated_imu) - np.array(y_real), label=f'Noisy - Real (MSE={mse_imu_estimated_position_no_filter:.2f})', color='orange')
    ax3.set_title('Difference: Noisy Measurements vs Real Data (IMU)')
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Difference')
    ax3.legend(loc="upper left")

    # Second row, right: Difference filtered vs real
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.plot(x, np.array(y_filtered) - np.array(y_real), label=f'Filtered - Real (MSE={mse_position_with_filter:.2f})', color='green')
    ax4.set_title('Difference: Filtered Data vs Real Data')
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('Difference')
    ax4.legend(loc="upper left")
  
    # Second row, right: MSE comparison
    # ax3 = plt.subplot2grid((2, 2), (1, 1))
    # ax3.bar(['Position (No Filter)', 'Position (KF)'],
    #         [mse_position_no_filter, mse_position_with_filter],
    #         color=['orange', 'green'])
    # ax3.set_title('Mean Squared Error (MSE)')
    # ax3.set_ylabel('MSE')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
