"""
Animate how an MLP learns to separate non-linearly separable data.
Shows both the decision boundary in input space and the hidden representation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import optax

# Generate spiral dataset (more interesting than XOR)
def make_spirals(n_points=200, noise=0.15):
    np.random.seed(42)
    n = n_points // 2
    theta = np.linspace(0, 3 * np.pi, n)

    # Class 0: spiral going one way
    r0 = theta / (3 * np.pi)
    x0 = r0 * np.cos(theta) + noise * np.random.randn(n)
    y0 = r0 * np.sin(theta) + noise * np.random.randn(n)

    # Class 1: spiral going other way
    x1 = -r0 * np.cos(theta) + noise * np.random.randn(n)
    y1 = -r0 * np.sin(theta) + noise * np.random.randn(n)

    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.array([0] * n + [1] * n)

    return X.astype(np.float32), y.astype(np.float32)

# Simple MLP with one hidden layer
def mlp(params, x):
    W1, b1, W2, b2 = params
    h = jnp.tanh(x @ W1 + b1)  # Hidden layer
    return jax.nn.sigmoid(h @ W2 + b2).squeeze()  # Output

def get_hidden(params, x):
    W1, b1, W2, b2 = params
    return jnp.tanh(x @ W1 + b1)

def loss_fn(params, X, y):
    preds = mlp(params, X)
    return -jnp.mean(y * jnp.log(preds + 1e-7) + (1 - y) * jnp.log(1 - preds + 1e-7))

# Initialize
np.random.seed(0)
key = jax.random.PRNGKey(0)

hidden_dim = 16
params = [
    jax.random.normal(key, (2, hidden_dim)) * 0.5,  # W1
    jnp.zeros(hidden_dim),  # b1
    jax.random.normal(jax.random.PRNGKey(1), (hidden_dim, 1)) * 0.5,  # W2
    jnp.zeros(1),  # b2
]

X, y = make_spirals(300, noise=0.12)
optimizer = optax.adam(0.03)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, X, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Train and save snapshots
n_steps = 800
snapshot_every = 10
snapshots = []

for step in range(n_steps + 1):
    if step % snapshot_every == 0:
        snapshots.append([p.copy() for p in params])
    params, opt_state, loss = train_step(params, opt_state, X, y)

print(f"Final loss: {loss:.4f}")
print(f"Saved {len(snapshots)} snapshots")

# Compute fixed PCA projection from final model
H_final = np.array(get_hidden(snapshots[-1], X))
H_final_centered = H_final - H_final.mean(axis=0)
_, _, Vt_fixed = np.linalg.svd(H_final_centered, full_matrices=False)
Vt_fixed = Vt_fixed[:2]  # Fixed projection directions

# Compute fixed scale from final model
H_final_2d = H_final_centered @ Vt_fixed.T
fixed_scale = np.abs(H_final_2d).max() * 1.3

# Create animation
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Grid for decision boundary
xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 100), np.linspace(-1.3, 1.3, 100))
grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

# Colors
colors = np.array(['#4A90D9', '#D94A4A'])  # Blue and red

def update(frame):
    for ax in axes:
        ax.clear()

    p = snapshots[frame]
    step_num = frame * snapshot_every

    # Left: Decision boundary in input space
    Z = mlp(p, grid).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z, levels=np.linspace(0, 1, 20), cmap='RdBu_r', alpha=0.7)
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    axes[0].scatter(X[:, 0], X[:, 1], c=colors[y.astype(int)], s=20, edgecolors='white', linewidths=0.5)
    axes[0].set_xlim(-1.3, 1.3)
    axes[0].set_ylim(-1.3, 1.3)
    axes[0].set_aspect('equal')
    axes[0].set_title('Input space', fontsize=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Right: Hidden representation (project to 2D using fixed PCA from final model)
    H = np.array(get_hidden(p, X))
    H_centered = H - H_final.mean(axis=0)  # Use final model's mean for consistency
    H_2d = H_centered @ Vt_fixed.T

    axes[1].scatter(H_2d[:, 0], H_2d[:, 1], c=colors[y.astype(int)], s=20, edgecolors='white', linewidths=0.5)

    # Fixed scale for consistent view
    axes[1].set_xlim(-fixed_scale, fixed_scale)
    axes[1].set_ylim(-fixed_scale, fixed_scale)
    axes[1].set_aspect('equal')
    axes[1].set_title('Hidden rep. (PCA)', fontsize=12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(f'Step {step_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return axes

# Create animation
ani = FuncAnimation(fig, update, frames=len(snapshots), interval=80, blit=False)

# Save as GIF
output_path = '/Users/smsharma/Projects/ds595-ai4science/slides/figures/05-neural-networks/mlp_learning.gif'
ani.save(output_path, writer=PillowWriter(fps=12))
print(f"Saved animation to {output_path}")

plt.close()
