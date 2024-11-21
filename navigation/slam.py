import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, map_size):
        self.num_particles = num_particles
        self.particles = np.random.rand(num_particles, 3) * np.array([map_size[0], map_size[1], 2 * np.pi])
        self.weights = np.ones(num_particles) / num_particles

    def motion_update(self, delta, std_dev):
        """Оновлення частинок на основі моделі руху."""
        self.particles[:, 0] += delta[0] + np.random.normal(0, std_dev[0], self.num_particles)
        self.particles[:, 1] += delta[1] + np.random.normal(0, std_dev[1], self.num_particles)
        self.particles[:, 2] += delta[2] + np.random.normal(0, std_dev[2], self.num_particles)

    def sensor_update(self, measurements, map_data, sensor_std_dev):
        """Оновлення ваг частинок на основі сенсорних даних."""
        for i, particle in enumerate(self.particles):
            x, y, theta = particle
            # Симулюємо сенсорні вимірювання для цієї частинки
            predicted_measurements = simulate_sensor_reading(x, y, theta, map_data)
            error = np.linalg.norm(predicted_measurements - measurements)
            self.weights[i] = np.exp(-error**2 / (2 * sensor_std_dev**2))
        
        self.weights += 1.e-300  # Уникнення нулів
        self.weights /= sum(self.weights)  # Нормалізація

    def resample(self):
        """Перезразок частинок на основі їхніх ваг."""
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

def simulate_sensor_reading(x, y, theta, map_data):
    """Симуляція сенсора (проста функція для прикладу)."""
    return np.array([x, y])  # Повертаємо позицію як вимірювання

# Демонстрація
map_size = (100, 100)
filter = ParticleFilter(num_particles=1000, map_size=map_size)

# Модель руху
delta_motion = [1, 1, 0.1]
sensor_readings = [50, 50]  # Умовні дані сенсора
sensor_std_dev = 5

filter.motion_update(delta_motion, [0.5, 0.5, 0.1])
filter.sensor_update(sensor_readings, None, sensor_std_dev)
filter.resample()

print("Частинки після оновлення:")
print(filter.particles)
