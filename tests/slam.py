import numpy as np
import matplotlib.pyplot as plt

# Initialize state
def initialize_state(num_landmarks):
    state_dim = 3 + 2 * num_landmarks  # Robot pose (x, y, theta) + landmarks (x, y)
    state = np.zeros(state_dim)       # Initial state vector
    covariance = np.eye(state_dim) * 1e-3  # Small initial uncertainty
    return state, covariance

# Motion model
def motion_model(state, u, dt):
    x, y, theta = state[:3]
    v, w = u
    if np.abs(w) < 1e-6:  # Straight line motion
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta
    else:
        x_new = x + (-v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
        y_new = y + (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt)
        theta_new = theta + w * dt
    return np.array([x_new, y_new, theta_new])

def motion_jacobian(state, u, dt):
    """Compute the Jacobian of the motion model w.r.t the state."""
    x, y, theta = state[:3]
    v, w = u
    Fx_robot = np.eye(3)  # Jacobian for robot pose

    if np.abs(w) < 1e-6:  # Straight-line motion
        Fx_robot[0, 2] = -v * np.sin(theta) * dt
        Fx_robot[1, 2] = v * np.cos(theta) * dt
    else:  # Rotational motion
        Fx_robot[0, 2] = (-v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt)
        Fx_robot[1, 2] = (-v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)

    # Expand Fx to full state size
    state_dim = len(state)
    Fx = np.eye(state_dim)
    Fx[:3, :3] = Fx_robot

    return Fx

# Measurement model
def measurement_model(state, landmark_index):
    x, y, theta = state[:3]
    lx, ly = state[3 + 2 * landmark_index: 5 + 2 * landmark_index]
    dx, dy = lx - x, ly - y
    q = dx**2 + dy**2
    z_pred = np.array([np.sqrt(q), np.arctan2(dy, dx) - theta])  # Range, Bearing
    return z_pred

# Measurement Jacobian
def measurement_jacobian(state, landmark_index):
    x, y, theta = state[:3]
    lx, ly = state[3 + 2 * landmark_index: 5 + 2 * landmark_index]
    dx, dy = lx - x, ly - y
    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)

    H = np.zeros((2, len(state)))
    H[0, 0] = -dx / sqrt_q
    H[0, 1] = -dy / sqrt_q
    H[0, 3 + 2 * landmark_index] = dx / sqrt_q
    H[0, 4 + 2 * landmark_index] = dy / sqrt_q
    H[1, 0] = dy / q
    H[1, 1] = -dx / q
    H[1, 2] = -1
    H[1, 3 + 2 * landmark_index] = -dy / q
    H[1, 4 + 2 * landmark_index] = dx / q
    return H

# EKF prediction step
def ekf_predict(state, covariance, u, dt, motion_noise):
    """EKF prediction step."""
    Fx = motion_jacobian(state, u, dt)  # Full state Jacobian
    state[:3] = motion_model(state, u, dt)  # Predict robot pose
    covariance = Fx @ covariance @ Fx.T + np.pad(motion_noise, ((0, len(state) - 3), (0, len(state) - 3)))
    return state, covariance


# EKF update step
def ekf_update(state, covariance, z, landmark_index, measurement_noise):
    z_pred = measurement_model(state, landmark_index)
    H = measurement_jacobian(state, landmark_index)

    # Kalman gain
    S = H @ covariance @ H.T + measurement_noise
    K = covariance @ H.T @ np.linalg.inv(S)

    # Update state
    y = z - z_pred
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
    state += K @ y

    # Update covariance
    I = np.eye(len(state))
    covariance = (I - K @ H) @ covariance
    return state, covariance

# Main loop
def ekf_slam():
    num_landmarks = 5
    dt = 0.1  # Time step
    state, covariance = initialize_state(num_landmarks)

    # Define noises
    motion_noise = np.diag([0.1, 0.1, np.deg2rad(1)])**2
    measurement_noise = np.diag([0.5, np.deg2rad(5)])**2

    # Simulated inputs and measurements
    controls = [[1.0, 0.1] for _ in range(100)]  # Constant motion
    measurements = [np.array([3.0, np.deg2rad(45)]) for _ in range(100)]  # Fixed landmark

    for t, u in enumerate(controls):
        # Predict
        state, covariance = ekf_predict(state, covariance, u, dt, motion_noise)

        # Update for each landmark
        for i in range(num_landmarks):
            z = measurements[t]  # Simulated measurement
            state, covariance = ekf_update(state, covariance, z, i, measurement_noise)

    return state, covariance

# Import the EKF functions from the previous implementation

# Simulated environment setup
def generate_simulated_data(num_steps, num_landmarks, dt):
    """Generates ground truth trajectory and landmark positions."""
    # Define landmarks
    landmarks = np.random.uniform(-10, 10, (num_landmarks, 2))

    # Define robot trajectory
    robot_trajectory = []
    state = np.array([0, 0, 0])  # Initial pose (x, y, theta)
    for step in range(num_steps):
        v = 1.0  # Linear velocity
        w = 0.1 if step < num_steps // 2 else -0.1  # Angular velocity
        state = motion_model(state, [v, w], dt)
        robot_trajectory.append(state)

    robot_trajectory = np.array(robot_trajectory)
    return robot_trajectory, landmarks

def simulate_measurements(robot_trajectory, landmarks, noise_std):
    """Simulates range and bearing measurements to landmarks."""
    measurements = []
    for state in robot_trajectory:
        x, y, theta = state
        z = []
        for lx, ly in landmarks:
            dx, dy = lx - x, ly - y
            r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, noise_std[0])
            b = np.arctan2(dy, dx) - theta + np.random.normal(0, noise_std[1])
            z.append([r, b])
        measurements.append(z)
    return measurements

def plot_results(robot_trajectory, landmarks, estimated_states, step_interval=10):
    """Plots the ground truth and EKF-SLAM results."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ground truth
    ax.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], label="True Trajectory", linestyle="--", color="blue")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="red", label="True Landmarks", marker="x")

    # Estimated trajectory and landmarks
    estimated_trajectory = np.array([s[:3] for s in estimated_states])
    ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label="Estimated Trajectory", color="green")

    # Extract landmarks from the final estimated state
    estimated_landmarks = np.array([
        estimated_states[-1][3 + 2 * i: 3 + 2 * i + 2]
        for i in range(len(landmarks))
    ])
    ax.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], c="orange", marker="o", label="Estimated Landmarks")

    # Formatting
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend()
    ax.set_title("EKF-SLAM Visualization")
    plt.grid()
    plt.show()

# Main function
def main():
    num_steps = 100
    num_landmarks = 5
    dt = 0.1
    measurement_noise_std = [0.5, np.deg2rad(5)]

    # Generate simulated data
    robot_trajectory, landmarks = generate_simulated_data(num_steps, num_landmarks, dt)
    measurements = simulate_measurements(robot_trajectory, landmarks, measurement_noise_std)

    # Initialize EKF-SLAM
    state, covariance = initialize_state(num_landmarks)
    motion_noise = np.diag([0.1, 0.1, np.deg2rad(1)])**2
    measurement_noise = np.diag(measurement_noise_std)**2

    # Run EKF-SLAM
    estimated_states = []
    for t in range(num_steps):
        # Predict step
        u = [1.0, 0.1 if t < num_steps // 2 else -0.1]  # Simulated control inputs
        state, covariance = ekf_predict(state, covariance, u, dt, motion_noise)

        # Update step
        for i, z in enumerate(measurements[t]):
            state, covariance = ekf_update(state, covariance, z, i, measurement_noise)

        estimated_states.append(state.copy())

    # Plot results
    plot_results(robot_trajectory, landmarks, estimated_states)

if __name__ == "__main__":
    main()

