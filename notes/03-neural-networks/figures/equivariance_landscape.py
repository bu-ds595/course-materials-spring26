import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75

# Define the loss functions
def f(x):
    return 0.5 * np.sin(2 * x) + 0.3 * x**2

def g(x):
    return 0.4 * np.cos(1.5 * x + 1) + 0.2 * x**2

# Combined loss on the diagonal (equivariant constraint x=y)
def loss_1d(x):
    return f(x) + g(x)

# Unconstrained loss with soft constraint
def loss_2d(x, y, gamma=2.0):
    return f(x) + g(y) + gamma * np.abs(x - y)

fig = plt.figure(figsize=(DOUBLE_COL, 2.5))

# Left panel: 1D constrained optimization
ax1 = fig.add_subplot(1, 2, 1)
x = np.linspace(-2, 3, 500)
y_loss = loss_1d(x)

ax1.plot(x, y_loss, 'k-', linewidth=1.2)

# Optimization path (gradient descent from left)
x_start = -1.8
path_x = [x_start]
lr = 0.15
for _ in range(25):
    # Numerical gradient
    eps = 1e-5
    grad = (loss_1d(path_x[-1] + eps) - loss_1d(path_x[-1] - eps)) / (2 * eps)
    new_x = path_x[-1] - lr * grad
    path_x.append(new_x)

path_y = [loss_1d(px) for px in path_x]
ax1.plot(path_x, path_y, 'o-', color='#CC3311', markersize=3, linewidth=0.8, alpha=0.8)

ax1.set_xlabel('$x$')
ax1.set_ylabel('loss')
ax1.set_title('$f(x) + g(x)$', fontsize=9)
ax1.set_xlim(-2, 3)

# Right panel: 2D surface with constraint
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

x_grid = np.linspace(-2, 3, 80)
y_grid = np.linspace(-2, 3, 80)
X, Y = np.meshgrid(x_grid, y_grid)
Z = loss_2d(X, Y, gamma=3.0)

# Plot surface
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0.1, 
                  edgecolor='white', rcount=40, ccount=40)

# Plot the diagonal constraint x=y
diag = np.linspace(-2, 3, 100)
z_diag = loss_2d(diag, diag, gamma=3.0)
ax2.plot(diag, diag, z_diag, 'k-', linewidth=1.5, zorder=10)

# Optimization path on the diagonal
path_3d_x = path_x
path_3d_y = path_x  # constrained to x=y
path_3d_z = [loss_2d(px, px, gamma=3.0) for px in path_x]
ax2.plot(path_3d_x, path_3d_y, path_3d_z, 'o-', color='#CC3311', 
         markersize=2.5, linewidth=0.8, alpha=0.9, zorder=11)

ax2.set_xlabel('$x$', labelpad=-8)
ax2.set_ylabel('$y$', labelpad=-8)
ax2.set_zlabel('loss', labelpad=-8)
ax2.set_title('$f(x) + g(y) + \gamma|x-y|$', fontsize=9)
ax2.view_init(elev=25, azim=-50)
ax2.set_xlim(-2, 3)
ax2.set_ylim(-2, 3)

# Reduce tick label size for 3D
ax2.tick_params(axis='both', which='major', labelsize=6, pad=-3)

plt.tight_layout()
plt.savefig('equivariance_landscape.pdf', bbox_inches='tight')
plt.savefig('equivariance_landscape.png', dpi=150, bbox_inches='tight')
print("Saved equivariance_landscape.pdf")
