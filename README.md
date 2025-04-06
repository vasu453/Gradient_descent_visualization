import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

# Define the gradient of the function
def gradient(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Set random seed for reproducibility
np.random.seed(42)

# Initialize random starting point
start_point = np.random.uniform(-10, 10, 2)

# Gradient descent parameters
learning_rate = 0.1
iterations = 50

# Store the path taken by gradient descent
path = [start_point]
current_point = start_point

# Perform gradient descent
for _ in range(iterations):
    grad = gradient(*current_point)
    next_point = current_point - learning_rate * grad
    path.append(next_point)
    current_point = next_point

path = np.array(path)

# Create a plot showing the contour of f(x, y) and the path taken by gradient descent
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 6))

# Plot the contour of the function
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x, y) = x² + y²')

# Plot the path taken by gradient descent
plt.plot(path[:, 0], path[:, 1], 'r-o', label='Gradient Descent Path')

# Mark the starting and ending points
plt.scatter(path[0, 0], path[0, 1], color='blue', label='Start Point', zorder=5)
plt.scatter(path[-1, 0], path[-1, 1], color='green', label='Minimum Point', zorder=5)

plt.title('Gradient Descent on f(x, y) = x² + y²')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
