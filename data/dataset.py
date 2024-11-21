import cv2
import numpy

def map_image_to_array(image_path, threshold=128, scale_factor=0.1):
    """
    Converts a room map image to a binary array with reduced size.
    
    Args:
        image_path (str): Path to the image.
        threshold (int): Threshold for binarization (default 128).
        scale_factor (float): Scaling factor to reduce the size (default 0.1).
    
    Returns:
        np.ndarray: Binary array where 1 is a passage, 0 is an obstacle.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load the image. Check the path.")

    # Resize the image to reduce resolution
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Binarize the image
    _, binary_image = cv2.threshold(resized_image, threshold, 1, cv2.THRESH_BINARY)

    # Invert the binary map (1 = passage, 0 = obstacle)
    # binary_map = 1 - binary_image
    binary_map = binary_image
    
    return binary_map

image_path = './map.jpg'  # Шлях до картинки
binary_map = map_image_to_array(image_path)

print("Бінаризована карта приміщення:")
print(binary_map)

a = numpy.array(binary_map)
unique, counts = numpy.unique(a, return_counts=True)

print(dict(zip(unique, counts)))
