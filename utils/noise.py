import numpy as np

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
