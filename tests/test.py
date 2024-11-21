import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Kalman Filter setup (assuming 2D position and velocity)
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.R *= 0.5  # Measurement noise covariance
    kf.Q *= 0.1  # Process noise covariance
    kf.P *= 1    # Initial uncertainty
    return kf

# Neural Network for optimizing Kalman filter usage
def create_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output: single parameter adjustment factor
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to perform SLAM (hypothetical placeholder for multi-sensor SLAM)
def slam_update(gps_data, sensor_data):
    # Combine GPS and sensor data into a 2D position update (example logic)
    position = (gps_data + sensor_data) / 2  # Simplified fusion
    return position

# Function to predict Kalman filter adjustments using neural network
def adjust_kalman_params(model, gps_data, sensor_data):
    input_data = np.concatenate((gps_data, sensor_data)).reshape(1, -1)
    adjustment_factor = model.predict(input_data, verbose=0)[0, 0]
    # Ensure the adjustment factor is reasonable
    adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)
    return adjustment_factor

# Main navigation loop
def navigation_loop(true_positions, gps_data_stream, sensor_data_stream, model, kf):
    positions = []
    kf_positions = []
    slam_positions = []
    for i, (true_pos, gps_data, sensor_data) in enumerate(zip(true_positions, gps_data_stream, sensor_data_stream)):
        # SLAM Update: Fuse GPS and sensor data to estimate current position
        slam_position = slam_update(gps_data, sensor_data)
        slam_positions.append(slam_position)
        
        # Predict: Kalman filter prediction step
        kf.predict()
        
        # Adjust Kalman Filter parameters dynamically
        adjustment_factor = adjust_kalman_params(model, gps_data, sensor_data)
        kf.R *= adjustment_factor  # Adjust measurement noise
        kf.Q *= adjustment_factor  # Adjust process noise
        
        # Update Kalman filter with SLAM data
        kf.update(slam_position)
        
        # Store the updated position estimate
        positions.append(kf.x[:2])
        kf_positions.append(kf.x[:2].copy())
        
        # Optionally reset adjustment factors to default
        kf.R /= adjustment_factor
        kf.Q /= adjustment_factor
    
    return positions, kf_positions, slam_positions

# Simulation of true positions, GPS, and sensor data streams (for testing)
np.random.seed(42)  # For reproducibility

time_steps = 50
true_positions = [np.array([i, i]) for i in range(time_steps)]
gps_noise_std = 0.5
sensor_noise_std = 0.3

gps_data_stream = [pos + np.random.normal(0, gps_noise_std, 2) for pos in true_positions]
sensor_data_stream = [pos + np.random.normal(0, sensor_noise_std, 2) for pos in true_positions]

# Initialize Kalman Filter and Neural Network model
kf = create_kalman_filter()
model = create_nn_model(input_shape=4)

# For demonstration, we train the NN model on random data (since we lack real training data)
# In practice, the model should be trained on historical data
train_X = np.random.rand(1000, 4)
train_y = np.random.rand(1000, 1)
model.fit(train_X, train_y, epochs=5, verbose=0)

# Run navigation loop
positions, kf_positions, slam_positions = navigation_loop(true_positions, gps_data_stream, sensor_data_stream, model, kf)

# Convert list of arrays to arrays for plotting
true_positions = np.array(true_positions)
gps_data_stream = np.array(gps_data_stream)
sensor_data_stream = np.array(sensor_data_stream)
kf_positions = np.array(kf_positions)
slam_positions = np.array(slam_positions)

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'k-', label='True Position')
plt.scatter(gps_data_stream[:, 0], gps_data_stream[:, 1], c='r', label='GPS Measurements', alpha=0.6)
plt.scatter(sensor_data_stream[:, 0], sensor_data_stream[:, 1], c='g', label='Sensor Measurements', alpha=0.6)
plt.plot(slam_positions[:, 0], slam_positions[:, 1], 'b--', label='SLAM Output')
plt.plot(kf_positions[:, 0], kf_positions[:, 1], 'm-', label='Kalman Filter Estimate')
plt.legend()
plt.title('Navigation Path with Sensor Fusion and Kalman Filter')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.show()
