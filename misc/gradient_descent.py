import numpy as np
import matplotlib.pyplot as plt

# Define a simple quadratic function: f(x) = (x-3)^2 + 2
f = lambda x: (x-3)**2 + 2
df = lambda x: 2*(x-3)  # Derivative

# Gradient descent parameters
x = 8.0  # Start far from the minimum
learning_rate = 0.2
steps = 10

x_history = [x]
for i in range(steps):
    grad = df(x)
    x = x - learning_rate * grad
    x_history.append(x)

# Plotting
x_vals = np.linspace(0, 10, 100)
y_vals = f(x_vals)

plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, label='f(x) = (x-3)^2 + 2')
plt.scatter(x_history, [f(xi) for xi in x_history], color='red', zorder=5, label='Gradient Descent Steps')
plt.plot(x_history, [f(xi) for xi in x_history], color='red', linestyle='--', alpha=0.5)
for i, (xi, yi) in enumerate(zip(x_history, [f(xi) for xi in x_history])):
    plt.text(xi, yi+0.5, str(i), fontsize=8, ha='center')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Example')
plt.legend()
plt.grid(True)
plt.show()