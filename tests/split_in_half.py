import os
import cv2
import glob

path = "clean"
os.makedirs(path, exist_ok=True)

for filename in glob.glob('stereophoto.jpg'):
    
    img = cv2.imread(filename)
    height, width, depth = img.shape
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]

    cv2.imwrite(f'left_image.jpg', s1)
    cv2.imwrite(f'right_image.jpg', s2)