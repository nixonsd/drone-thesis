import cv2
import numpy as np

def rectify_and_compute_depth(left_image_path, right_image_path, calibration_params):
  """
  Rectifies stereo images and computes a depth map using precomputed calibration parameters.

  :param left_image_path: Path to the left image.
  :param right_image_path: Path to the right image.
  :param calibration_params: Dictionary containing stereo camera calibration parameters.
  :return: Rectified left and right images, depth map, and normalized disparity map.
  """
  # Load left and right images
  left_image = cv2.imread(left_image_path)
  right_image = cv2.imread(right_image_path)

  # Extract calibration parameters and ensure they are float64
  camera_matrix_left = np.array(calibration_params['camera_matrix_left'], dtype=np.float64)
  dist_coeffs_left = np.array(calibration_params['dist_coeffs_left'], dtype=np.float64)
  camera_matrix_right = np.array(calibration_params['camera_matrix_right'], dtype=np.float64)
  dist_coeffs_right = np.array(calibration_params['dist_coeffs_right'], dtype=np.float64)
  R = np.array(calibration_params['R'], dtype=np.float64)
  T = np.array(calibration_params['T'], dtype=np.float64)
  image_size = calibration_params['image_size']

  # Stereo rectification
  R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
  )

  # Compute rectification maps
  map1x, map1y = cv2.initUndistortRectifyMap(
    camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_32FC1
  )
  map2x, map2y = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_32FC1
  )

  # Apply rectification
  rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
  rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

  # Compute depth map
  stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # Must be divisible by 16
    blockSize=15,
    P1=8 * 3 * 15**2,
    P2=32 * 3 * 15**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=63
  )
  disparity = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

  # Normalize the disparity map for visualization
  disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  disparity_normalized = np.uint8(disparity_normalized)

  return rectified_left, rectified_right, disparity, disparity_normalized


def main():
  # Stereo calibration parameters (replace with actual values from your source)
  calibration_params = {
    "camera_matrix_left": np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]]),
    "dist_coeffs_left": np.zeros(5),  # Assume no distortion
    "camera_matrix_right": np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]]),
    "dist_coeffs_right": np.zeros(5),  # Assume no distortion
    "R": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Assume cameras are aligned
    "T": np.array([-0.1, 0, 0]),  # Translation vector (baseline = 0.1 meters)
    "image_size": (1280, 720)  # Image resolution
  }

  # Paths to the left and right images
  left_image_path = "left_image.jpg"
  right_image_path = "right_image.jpg"

  # Rectify and compute depth
  rectified_left, rectified_right, disparity, disparity_normalized = rectify_and_compute_depth(
    left_image_path, right_image_path, calibration_params
  )

  # Display results
  cv2.imshow("Rectified Left", rectified_left)
  cv2.imshow("Rectified Right", rectified_right)
  cv2.imshow("Depth Map (Normalized)", disparity_normalized)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
