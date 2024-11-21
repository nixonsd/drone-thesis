import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Встановлення випадкового зерна для відтворюваності результатів
np.random.seed(42)

# Параметри симуляції
dt = 0.1  # крок часу
t = np.arange(0, 20, dt)  # часовий вектор

# Істинна траєкторія дрона (прямолінійний рух з синусоїдальними коливаннями)
true_x = t
true_y = np.sin(t)

# Генерація шумних вимірювань GPS
gps_noise_std = 0.5
gps_measurements = np.vstack((
    true_x + np.random.normal(0, gps_noise_std, size=t.shape),
    true_y + np.random.normal(0, gps_noise_std, size=t.shape)
)).T

# Генерація шумних вимірювань SLAM
slam_noise_std = 0.2
slam_measurements = np.vstack((
    true_x + np.random.normal(0, slam_noise_std, size=t.shape),
    true_y + np.random.normal(0, slam_noise_std, size=t.shape)
)).T

# Налаштування Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0., 0., 1., 0.])  # початковий стан [x, y, vx, vy]
kf.F = np.array([[1, 0, dt, 0],    # матриця переходу стану
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],     # матриця спостережень
                 [0, 1, 0, 0]])
kf.P *= 500                         # початкова коваріація
kf.R = np.eye(2) * 0.1              # коваріація шуму вимірювань
kf.Q = np.eye(4) * 0.01             # шум процесу

# Масиви для збереження результатів
estimated_positions = []

# Симуляція
for i in range(len(t)):
    # Злиття вимірювань GPS та SLAM
    measurement = (gps_measurements[i] + slam_measurements[i]) / 2
    
    # Оновлення Kalman Filter
    kf.predict()
    kf.update(measurement)
    
    # Збереження оцінених позицій
    estimated_positions.append(kf.x[:2])

# Перетворення результатів у зручний формат
estimated_positions = np.array(estimated_positions)

# Побудова графіків
plt.figure(figsize=(12, 8))
plt.plot(true_x, true_y, label='Істинна траєкторія', linewidth=2)
plt.scatter(gps_measurements[:, 0], gps_measurements[:, 1], color='red', alpha=0.5, label='GPS вимірювання')
plt.scatter(slam_measurements[:, 0], slam_measurements[:, 1], color='green', alpha=0.5, label='SLAM вимірювання')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], color='black', linestyle='--', label='Оцінка Kalman Filter', linewidth=2)
plt.title('Симуляція руху дрона з об’єднанням даних GPS та SLAM')
plt.xlabel('X позиція')
plt.ylabel('Y позиція')
plt.legend()
plt.grid(True)
plt.show()
