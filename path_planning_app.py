import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from navigation.path_planning import PathPlanning
from data.dataset import map_image_to_array
import random
import math

class PathPlanningApp:
    def __init__(self, image_path, threshold=128, scale_factor=0.8, min_distance=5):
        self.image_path = image_path
        self.grid = map_image_to_array(image_path, threshold=threshold, scale_factor=scale_factor)
        self.planner = PathPlanning(row=len(self.grid), col=len(self.grid[0]))
        self.min_distance = min_distance
        self.path = []
        self.src = None
        self.dest = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.refresh_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
        self.refresh_button = Button(self.refresh_button_ax, 'Refresh')
        self.refresh_button.on_clicked(self.refresh)
        self.run_pathfinding()

    def pick_random_points(self):
        valid_points = [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) if self.grid[i][j] == 1]
        while True:
            src = random.choice(valid_points)
            dest = random.choice(valid_points)
            if src != dest:
                distance = math.sqrt((src[0] - dest[0]) ** 2 + (src[1] - dest[1]) ** 2)
                if distance >= self.min_distance:
                    return src, dest

    def custom_trace_path(self, cell_details, dest):
        row, col = dest
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            self.path.append((row, col))
            row, col = cell_details[row][col].parent_i, cell_details[row][col].parent_j
        self.path.append((row, col))
        self.path.reverse()

    def run_pathfinding(self):
        self.src, self.dest = self.pick_random_points()
        self.path = []
        self.planner.trace_path = self.custom_trace_path
        self.planner.a_star_search(self.grid, self.src, self.dest)
        self.display_path()

    def display_path(self):
        self.ax.clear()
        grid_display = [[1 if cell == 1 else 0 for cell in row] for row in self.grid]
        self.ax.imshow(grid_display, cmap='gray', origin='upper')

        if self.path:
            y, x = zip(*self.path)
            self.ax.plot(x, y, marker='o', color='red', linewidth=1)

        self.ax.scatter(self.src[1], self.src[0], color='green', label='Start', zorder=5)
        self.ax.scatter(self.dest[1], self.dest[0], color='blue', label='Destination', zorder=5)
        self.ax.legend()
        self.ax.set_title("Path Planning Visualization")
        plt.draw()

    def refresh(self, event):
        self.run_pathfinding()

if __name__ == "__main__":
    app = PathPlanningApp(image_path='map.jpg', threshold=128, scale_factor=0.5, min_distance=4)
    plt.show()
