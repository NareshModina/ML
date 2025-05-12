import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x1, x2) = x1^2 + x2^2
def f(x1, x2):
    return x1**2 + x2**2

# Define the gradient of f(x1, x2)
def gradient(x1, x2):
    df_dx1 = 2 * x1
    df_dx2 = 2 * x2
    return np.array([df_dx1, df_dx2])

# Gradient descent algorithm for two variables
def gradient_descent_2d(starting_point, learning_rate, num_iterations):
    x = np.array(starting_point, dtype=float)
    x_history = [x.copy()]  # Store a copy of the array

    for _ in range(num_iterations):
        grad = gradient(x[0], x[1])
        x = x - learning_rate * grad
        x_history.append(x.copy())

    return np.array(x_history)

# Parameters
starting_point = [4.0, 4.0]  # Starting point (x1, x2)
learning_rate = 0.1  # Step size
num_iterations = 30  # Number of iterations

# Run gradient descent
x_history = gradient_descent_2d(starting_point, learning_rate, num_iterations)
z_history = [f(x[0], x[1]) for x in x_history]

# Generate points for plotting the surface
num_points = 50
x1 = np.linspace(-5, 5, num_points)
x2 = np.linspace(-5, 5, num_points)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Set up the plot
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
surface = ax.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.7)
path, = ax.plot([], [], [], 'ro-', label='Gradient Descent Path')
point, = ax.plot([], [], [], 'go', markersize=8, label='Current Point')
ax.set_title(f'Gradient Descent for f(x1, x2) = x1^2 + x2^2 (Learning Rate = {learning_rate})', color='white')
ax.set_xlabel('x1', color='white')
ax.set_ylabel('x2', color='white')
ax.set_zlabel('f(x1, x2)', color='white')
ax.grid(True, color='gray')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Initialization function for animation
def init():
    path.set_data([], [])
    path.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return path, point,

# Animation update function
def update(frame):
    x_data = x_history[:frame+1, 0]
    y_data = x_history[:frame+1, 1]
    z_data = z_history[:frame+1]
    path.set_data(x_data, y_data)
    path.set_3d_properties(z_data)
    point.set_data([x_history[frame, 0]], [x_history[frame, 1]])
    point.set_3d_properties([z_history[frame]])
    return path, point,

# Create animation
ani = FuncAnimation(fig, update, frames=len(x_history), init_func=init, blit=True, interval=750)

# To display the animation
plt.show()

# To save the animation (requires a writer like 'pillow' or 'ffmpeg')
# ani.save('gradient_descent_2d_animation.gif', writer='pillow')
# ani.save('gradient_descent_2d_animation.mp4', writer='ffmpeg')