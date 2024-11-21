import math
import heapq

class Cell:
    def __init__(self):
        # Індекси батьківської комірки
        self.parent_i = 0  
        self.parent_j = 0  
        
        # Загальна вартість комірки (f = g + h)
        self.f = float('inf')  
        
        # Вартість від початкової до поточної комірки
        self.g = float('inf')  
        
        # Оціночна вартість від поточної комірки до цілі
        self.h = 0  

class PathPlanning:
    def __init__(self, row=9, col=10):
        # Розміри сітки
        self.ROW = row
        self.COL = col

    def is_valid(self, row, col):
        # Перевірка, чи знаходиться комірка в межах сітки
        return (0 <= row < self.ROW) and (0 <= col < self.COL)

    def is_unblocked(self, grid, row, col):
        # Перевірка, чи є комірка прохідною (1 — прохідна, 0 — непрохідна)
        return grid[row][col] == 1

    def is_destination(self, row, col, dest):
        # Перевірка, чи є поточна комірка цільовою
        return row == dest[0] and col == dest[1]

    def calculate_h_value(self, row, col, dest):
        # Обчислення евристичної вартості (Євклідова відстань до цілі)
        return math.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

    def trace_path(self, cell_details, dest):
        # Відслідковування шляху від цілі до початкової точки
        print("Шлях:")
        path = []
        row, col = dest

        # Прямуємо від цілі до початкової точки, використовуючи батьківські комірки
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            row, col = cell_details[row][col].parent_i, cell_details[row][col].parent_j

        # Додаємо початкову комірку та виводимо шлях
        path.append((row, col))
        path.reverse()
        for pos in path:
            print("->", pos, end=" ")
        print()

    def a_star_search(self, grid, src, dest):
        # Перевірка валідності початкової та цільової комірки
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            print("Неправильна початкова або цільова комірка")
            return

        if not self.is_unblocked(grid, src[0], src[1]) or not self.is_unblocked(grid, dest[0], dest[1]):
            print("Початкова або цільова комірка заблокована")
            return

        if self.is_destination(src[0], src[1], dest):
            print("Ви вже на цільовій комірці")
            return

        # Ініціалізація списків комірок
        closed_list = [[False] * self.COL for _ in range(self.ROW)]
        cell_details = [[Cell() for _ in range(self.COL)] for _ in range(self.ROW)]

        # Ініціалізація початкової комірки
        i, j = src
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i, cell_details[i][j].parent_j = i, j

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        found_dest = False

        # Основний цикл алгоритму A*
        while open_list:
            _, i, j = heapq.heappop(open_list)
            closed_list[i][j] = True

            # Перевірка сусідніх комірок у всіх напрямках
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                new_i, new_j = i + direction[0], j + direction[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    if self.is_destination(new_i, new_j, dest):
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("Цільову комірку знайдено")
                        self.trace_path(cell_details, dest)
                        found_dest = True
                        return

                    # Обчислення вартостей g, h та f для сусідньої комірки
                    g_new = cell_details[i][j].g + 1.0
                    h_new = self.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # Оновлення комірки, якщо нова вартість f менша
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

        if not found_dest:
            print("Не вдалося знайти шлях до цілі")

# Тестовий приклад для запуску алгоритму
if __name__ == "__main__":          
    # Приклад сітки (1 - прохідна комірка, 0 - заблокована комірка)
    grid = [
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
    ]

    # Початкова та цільова комірки
    src = [8, 0]
    dest = [0, 0]

    # Виконання пошуку A*
    planner = PathPlanning()
    planner.a_star_search(grid, src, dest)