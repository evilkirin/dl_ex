import numpy as np
import matplotlib.pyplot as plt

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))


# plt.interactive(False)
# Plot the red and blue points
plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')


# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

plt.show(block=True)