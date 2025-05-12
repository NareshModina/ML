import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function f(x) = x^2
def f(x):
    return x**2

# Define the derivative of f(x), f'(x) = 2x
def df(x):
    return 2*x

# Gradient descent algorithm
def gradient_descent(starting_x, learning_rate, num_iterations):
    x = starting_x
    x_history = [x]

    for _ in range(num_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)

    return x_history

# Parameters
starting_x = 4.0  # Starting point
learning_rate = 0.1  # Step size
num_iterations = 20  # Number of iterations

# Run gradient descent
x_history = gradient_descent(starting_x, learning_rate, num_iterations)

# Generate points for plotting f(x) = x^2
x_values = np.linspace(-5, 5, 100)
y_values = f(x_values)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
ax.set_facecolor('black')
line, = ax.plot([], [], 'ro-', label='Gradient Descent Path')
point, = ax.plot([], [], 'yo', markersize=10, label='Current Point')
ax.plot(x_values, y_values, 'w-', label='f(x) = x^2')
ax.set_title(f'Gradient Descent for f(x) = x^2 (Learning Rate = {learning_rate})', color='white')
ax.set_xlabel('x', color='white')
ax.set_ylabel('f(x)', color='white')
ax.grid(True, color='gray')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Initialization function for animation
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point,

# Animation update function
def update(frame):
    x_data = x_history[:frame+1]
    y_data = [f(x) for x in x_data]
    line.set_data(x_data, y_data)
    point.set_data(x_history[frame], f(x_history[frame]))
    return line, point,

# Create animation
ani = FuncAnimation(fig, update, frames=len(x_history), init_func=init, blit=True, interval=750)

# To display the animation within a compatible environment (like Jupyter Notebook)
plt.show()

# If you still want to save a static image of the final state:
ani.save('gradient_descent_final.png')