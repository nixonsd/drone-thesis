import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_depth_map_with_bm(left_image, right_image, focal_length, baseline, num_disparities=16 * 10, block_size=15):
  """
  Computes a depth map using the StereoBM algorithm and calculates the minimal distance.

  :param left_image: Grayscale image from the left camera.
  :param right_image: Grayscale image from the right camera.
  :param focal_length: Focal length of the stereo cameras in pixels.
  :param baseline: Distance between the two cameras in meters.
  :param num_disparities: Maximum disparity minus minimum disparity. Must be divisible by 16.
  :param block_size: Size of the matching block. Must be an odd number >= 5.
  :return: Disparity map (disparity map), normalized disparity map, and minimal distance.
  """
  # Validate inputs
  if len(left_image.shape) != 2 or len(right_image.shape) != 2:
    raise ValueError("Both left_image and right_image must be grayscale (2D arrays).")

  # Create StereoBM object
  stereo = cv2.StereoBM_create(
    numDisparities=num_disparities,  # Number of disparities (must be divisible by 16)
    blockSize=block_size  # Size of the matching block
  )

  # Preprocess images with histogram equalization
  left_image = cv2.equalizeHist(left_image)
  right_image = cv2.equalizeHist(right_image)

  # Compute disparity map
  disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

  # Compute depth from disparity
  depth = np.zeros_like(disparity, dtype=np.float32)
  valid_disparity_mask = disparity > 0  # Disparity must be positive
  depth[valid_disparity_mask] = (focal_length * baseline) / disparity[valid_disparity_mask]

  # Calculate minimal distance
  min_distance = np.min(depth[valid_disparity_mask]) if np.any(valid_disparity_mask) else None

  # Normalize the disparity map for visualization
  disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  disparity_normalized = np.uint8(disparity_normalized)

  return disparity, disparity_normalized, min_distance


def display_results_with_distance(left_image, right_image, disparity_normalized, min_distance):
  """
  Displays the left image, right image, and disparity map with a gradient representation.

  :param left_image: Grayscale image from the left camera.
  :param right_image: Grayscale image from the right camera.
  :param disparity_normalized: Normalized disparity map for visualization.
  :param min_distance: Minimal distance computed from the depth map.
  """
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  
  # Display the left image
  axes[0].imshow(left_image, cmap="gray")
  axes[0].set_title("Left Image")
  axes[0].axis("off")

  # Display the right image
  axes[1].imshow(right_image, cmap="gray")
  axes[1].set_title("Right Image")
  axes[1].axis("off")

  # Display the disparity map with a gradient (plasma colormap)
  disparity_plot = axes[2].imshow(disparity_normalized, cmap="plasma")
  axes[2].set_title(f"Depth Map (Min Distance: {min_distance:.2f}m)" if min_distance else "Depth Map")
  axes[2].axis("off")

  # Add a color bar for the disparity map
  fig.colorbar(disparity_plot, ax=axes[2], fraction=0.03, pad=0.04, label="Relative Depth")

  plt.tight_layout()
  plt.show()


def main():
  # Load the rectified left and right images in grayscale
  left_image_path = "./left_image.ppm"
  right_image_path = "./right_image.ppm"

  left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
  right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

  if left_image is None or right_image is None:
    raise FileNotFoundError("Could not load left or right images. Check the file paths.")

  # Camera parameters
  # Assuming Canon PowerShot A620 7.1MP Digital Camera
  focal_length = 3124  # Example focal length in pixels
  baseline = 0.12  # Example baseline in meters (distance between cameras)

  # Compute the depth map using StereoBM
  num_disparities = 128  # Must be divisible by 16
  block_size = 15  # Block size for matching
  disparity, disparity_normalized, min_distance = compute_depth_map_with_bm(
    left_image, right_image, focal_length, baseline, num_disparities, block_size
  )

  # Display the results using Matplotlib
  display_results_with_distance(left_image, right_image, disparity_normalized, min_distance)


if __name__ == "__main__":
  main()
