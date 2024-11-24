from sympy import symbols
from sympy.matrices import Matrix

x0, v0, dt = symbols('x0 v0 dt')
X = Matrix([[x0], [v0]])
A = Matrix([[1, dt], [0, 1]])

print(A * X)  # Correct order for matrix multiplication

Pk = Matrix([[1, 0], [0, 1]])
print(A * Pk)
