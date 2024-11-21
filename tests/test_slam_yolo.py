from data.dataset import map_image_to_array
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # Correct YOLO import

# Simulated SLAM system
class SimulatedSLAM:
    def __init__(self, grid_map):
        self.grid_map = np.array(grid_map)  # Initial grid map

    def update_map(self, new_obstacles):
        """Update the SLAM map with new obstacles."""
        for x, y in new_obstacles:
            if 0 <= x < self.grid_map.shape[0] and 0 <= y < self.grid_map.shape[1]:
                self.grid_map[x, y] = 1  # Mark as obstacle

    def visualize_map(self):
        """Visualize the current grid map."""
        plt.imshow(self.grid_map, cmap="gray", origin="upper")
        plt.title("SLAM Grid Map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.colorbar(label="0 = Free Space, 1 = Obstacle")
        plt.show()

# Simulated SLAM + YOLO integration
class SLAMWithYOLO:
    def __init__(self, grid_map):
        self.slam = SimulatedSLAM(grid_map)
        self.yolo = YOLO("yolov5s.pt")  # Load YOLOv5s model (pretrained)

    def simulate_frame(self, frame, dynamic_objects):
        """Simulate detection of objects in a single frame."""
        results = self.yolo.predict(frame)
        new_obstacles = []

        for box in results[0].boxes:
            label = box.cls  # Class label
            x1, y1, x2, y2 = box.xyxy  # Bounding box
            if label in dynamic_objects:
                print(f"Ignoring dynamic object: {label}")
            else:
                # Assume center of bounding box as obstacle location
                obstacle_x = int((x1 + x2) // 2)
                obstacle_y = int((y1 + y2) // 2)
                new_obstacles.append((obstacle_x, obstacle_y))

        self.slam.update_map(new_obstacles)

    def simulate_frames(self, num_frames, dynamic_objects):
        """Simulate multiple frames."""
        for frame_id in range(num_frames):
            print(f"Processing frame {frame_id + 1}/{num_frames}...")
            frame = self.generate_random_frame()
            self.simulate_frame(frame, dynamic_objects)
            self.slam.visualize_map()

    def generate_random_frame(self):
        """Generate a random frame for simulation."""
        # Simulated frame size (matches grid dimensions)
        frame = np.random.randint(0, 255, (self.slam.grid_map.shape[0], self.slam.grid_map.shape[1], 3), dtype=np.uint8)
        return frame

# Simulated binary grid map
# grid_map = [
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

image_path = './data/map.jpg'  # Шлях до картинки
binary_map = map_image_to_array(image_path)

# Test SLAM + YOLO system
slam_yolo_system = SLAMWithYOLO(binary_map)
slam_yolo_system.slam.visualize_map()  # Visualize initial map
slam_yolo_system.simulate_frames(num_frames=5, dynamic_objects=["person", "car"])  # Simulate with dynamic objects
