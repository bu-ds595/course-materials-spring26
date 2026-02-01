"""
Figure showing how neural networks learn nonlinear decision boundaries.
Demonstrates the power of composing linear transformations with nonlinearities.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'gray': '#BBBBBB',
}

# Set random seed for reproducibility
np.random.seed(42)

# Generate XOR-like data
n_points = 50
# Class 0: top-left and bottom-right
x0_1 = np.random.randn(n_points // 2, 2) * 0.3 + np.array([-0.7, 0.7])
x0_2 = np.random.randn(n_points // 2, 2) * 0.3 + np.array([0.7, -0.7])
x0 = np.vstack([x0_1, x0_2])

# Class 1: top-right and bottom-left
x1_1 = np.random.randn(n_points // 2, 2) * 0.3 + np.array([0.7, 0.7])
x1_2 = np.random.randn(n_points // 2, 2) * 0.3 + np.array([-0.7, -0.7])
x1 = np.vstack([x1_1, x1_2])

# Create meshgrid for decision boundaries
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Simple 1-hidden-layer network (manually tuned to show XOR-like boundary)
# This creates a piecewise linear boundary
def simple_network(x):
    # Hidden layer: 4 neurons
    W1 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]) * 2
    b1 = np.array([0, 0, 0, 0])
    h1 = relu(x @ W1.T + b1)

    # Output layer
    W2 = np.array([1, 1, -1, -1])
    b2 = 0.5
    out = sigmoid(h1 @ W2 + b2)
    return out

# Deeper network (smoother boundary)
def deep_network(x):
    # Layer 1
    W1 = np.array([[1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1, 0], [0, 1]])
    b1 = np.array([0, 0, 0, 0, 0, 0])
    h1 = relu(x @ W1.T + b1)

    # Layer 2
    W2 = np.array([[1, 1, -1, -1, 0.5, 0.5],
                   [-1, -1, 1, 1, -0.5, -0.5],
                   [0.5, 0.5, 0.5, 0.5, 1, -1],
                   [-0.5, -0.5, -0.5, -0.5, -1, 1]]) * 1.5
    b2 = np.zeros(4)
    h2 = relu(h1 @ W2.T + b2)

    # Output
    W3 = np.array([1, 1, -0.5, -0.5])
    b3 = -0.5
    out = sigmoid(h2 @ W3 + b3)
    return out

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.0))

# Panel (a): Data only - shows the problem
ax = axes[0]
ax.scatter(x0[:, 0], x0[:, 1], c=COLORS['blue'], s=15, alpha=0.7, label='Class 0')
ax.scatter(x1[:, 0], x1[:, 1], c=COLORS['orange'], s=15, alpha=0.7, label='Class 1')
# Show that linear boundary fails
ax.plot([-1.5, 1.5], [1.5, -1.5], '--', color=COLORS['gray'], linewidth=1.5, label='Linear')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('(a) Not linearly separable')
ax.set_aspect('equal')
ax.legend(loc='upper right', framealpha=0.9)

# Panel (b): Shallow network - piecewise linear boundary
ax = axes[1]
Z_simple = simple_network(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z_simple, levels=[0, 0.5, 1], colors=[COLORS['blue'], COLORS['orange']], alpha=0.25)
ax.contour(xx, yy, Z_simple, levels=[0.5], colors=['black'], linewidths=1.5)
ax.scatter(x0[:, 0], x0[:, 1], c=COLORS['blue'], s=15, alpha=0.7)
ax.scatter(x1[:, 0], x1[:, 1], c=COLORS['orange'], s=15, alpha=0.7)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$x_1$')
ax.set_title('(b) 1 hidden layer')
ax.set_aspect('equal')

# Panel (c): Deep network - smooth boundary
ax = axes[2]
Z_deep = deep_network(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z_deep, levels=[0, 0.5, 1], colors=[COLORS['blue'], COLORS['orange']], alpha=0.25)
ax.contour(xx, yy, Z_deep, levels=[0.5], colors=['black'], linewidths=1.5)
ax.scatter(x0[:, 0], x0[:, 1], c=COLORS['blue'], s=15, alpha=0.7)
ax.scatter(x1[:, 0], x1[:, 1], c=COLORS['orange'], s=15, alpha=0.7)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$x_1$')
ax.set_title('(c) 2 hidden layers')
ax.set_aspect('equal')

# Save
fig.savefig('nonlinear_decision_boundary.pdf', bbox_inches='tight')
plt.close()

print("Saved nonlinear_decision_boundary.pdf")
