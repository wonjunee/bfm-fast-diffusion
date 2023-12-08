import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Potential import Potential

# Parameters
a, b, c = 1, 1, 1

# Create a meshgrid of x and y values
x = np.linspace(0, 1, 400)
y = np.linspace(0, 1, 400)
x, y = np.meshgrid(x, y)

P = Potential(x, y)
#V = P.gaussian(0.5, 0.5, 0.01)
#V = P.double_well(0.25, 0.25, 0.75, 0.75, 100, 1)
V = P.trig(5, 3, 1)

#V = np.exp(-V)

# Compute the potential V(x, y)
#V = a * (x**2 - b**2)**2 + c * (y**2 - 1)**2

# Plot
fig, ax = plt.subplots(1, 1)
ax.contourf(x, y, V)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
