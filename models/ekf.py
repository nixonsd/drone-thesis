import numpy as np
from sympy import Matrix, symbols, lambdify

class ExtendedKalmanFilter:
    def __init__(self, state_dim, meas_dim, initial_state=None):
        """
        Initializes the Extended Kalman Filter.
        :param state_dim: Dimension of the state vector
        :param meas_dim: Dimension of the measurement vector
        :param initial_state: Initial state vector (optional)
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Initialize state vector
        self.x = initial_state if initial_state is not None else np.zeros((state_dim, 1))

        # Initialize covariance matrices
        self.P = np.eye(state_dim)  # State covariance matrix
        self.Q = np.eye(state_dim)  # Process noise covariance
        self.R = np.eye(meas_dim)  # Measurement noise covariance

    def compute_jacobian(self, func, state, *args):
        """
        Automatically computes the Jacobian matrix of a given function.
        :param func: Nonlinear function for which Jacobian is computed
        :param state: Current state vector
        :param args: Additional arguments required by the function
        :return: Jacobian matrix evaluated at the given state
        """
        # Create symbolic variables for the state vector
        symbols_list = symbols(f'x0:{self.state_dim}')
        symbolic_state = Matrix(symbols_list)

        # Symbolic expression for the function
        symbolic_func = Matrix(func(symbolic_state, *args))

        # Compute Jacobian symbolically
        jacobian_symbolic = symbolic_func.jacobian(symbolic_state)

        # Convert symbolic Jacobian to a numeric function
        jacobian_func = lambdify(symbols_list, jacobian_symbolic, 'numpy')

        # Evaluate the Jacobian at the current state
        return np.array(jacobian_func(*state.flatten()), dtype=np.float64)

    def predict(self, f, u=None):
        """
        Prediction step.
        :param f: Nonlinear state transition function
        :param u: Control input vector (optional)
        """
        # Compute the predicted state
        self.x = f(self.x, u)

        # Compute the Jacobian of the state transition function
        self.F = self.compute_jacobian(f, self.x, u)

        # Compute the predicted covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, h):
        """
        Update step.
        :param z: Measurement vector
        :param h: Nonlinear measurement function
        """
        # Compute the predicted measurement
        z_pred = h(self.x)

        # Compute the Jacobian of the measurement function
        self.H = self.compute_jacobian(h, self.x)

        # Compute the innovation (residual)
        y = z - z_pred

        # Compute the innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Compute the Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update the state estimate
        self.x = self.x + K @ y

        # Update the covariance estimate
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def add_dimension(self, initial_value, process_noise=1.0, measurement_noise=None):
        """
        Adds new dimensions for obstacles to the state vector and updates associated matrices.
        :param initial_value: List or array of initial values for the new state dimensions.
        Can be a flattened list [x1, y1, x2, y2, ...] or a single [x, y].
        :param process_noise: Process noise for the new dimensions.
        :param measurement_noise: Measurement noise for the new dimensions (optional).
        """
        # Ensure initial_value is a numpy array and reshape to (n, 1)
        initial_value = np.array(initial_value).reshape(-1, 1)

        # Update state dimension
        self.state_dim += initial_value.shape[0]

        # Expand state vector
        self.x = np.vstack([self.x, initial_value])

        # Expand state covariance matrix
        self.P = np.pad(self.P, ((0, initial_value.shape[0]), (0, initial_value.shape[0])), mode='constant', constant_values=0)
        np.fill_diagonal(self.P[-initial_value.shape[0]:, -initial_value.shape[0]:], process_noise)

        # Expand process noise covariance matrix
        self.Q = np.pad(self.Q, ((0, initial_value.shape[0]), (0, initial_value.shape[0])), mode='constant', constant_values=0)
        np.fill_diagonal(self.Q[-initial_value.shape[0]:, -initial_value.shape[0]:], process_noise)

        # Expand measurement noise covariance matrix if required
        if measurement_noise is not None:
            self.meas_dim += initial_value.shape[0]
            self.R = np.pad(self.R, ((0, initial_value.shape[0]), (0, initial_value.shape[0])), mode='constant', constant_values=0)
            np.fill_diagonal(self.R[-initial_value.shape[0]:, -initial_value.shape[0]:], measurement_noise)

    def set_process_noise(self, Q):
        """
        Sets the process noise covariance matrix.
        :param Q: Process noise covariance matrix
        """
        self.Q = Q

    def set_measurement_noise(self, R):
        """
        Sets the measurement noise covariance matrix.
        :param R: Measurement noise covariance matrix
        """
        self.R = R

    def get_state(self):
        """
        Returns the current state estimate.
        :return: State vector
        """
        return self.x

    def get_covariance(self):
        """
        Returns the current state covariance matrix.
        :return: State covariance matrix
        """
        return self.P
