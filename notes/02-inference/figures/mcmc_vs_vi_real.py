"""
Real MCMC vs VI comparison figure with actual optimization.
Shows evolution of VI approximation alongside ELBO trace.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load the publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

# Column widths (inches)
SINGLE_COL = 3.25
DOUBLE_COL = 6.75

# Colorblind-friendly Tol Vibrant palette
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'gray': '#BBBBBB',
    'cyan': '#33BBEE',
    'purple': '#AA3377',
}

np.random.seed(42)

# ============================================
# Define the banana posterior
# ============================================

def banana_log_prob(theta):
    """Log probability of banana distribution: p(x,y) ∝ exp(-x²/2 - (y - x²)²/2)"""
    x, y = theta[..., 0], theta[..., 1]
    return -0.5 * x**2 - 0.5 * (y - x**2)**2

def banana_density(x, y):
    """Unnormalized density for plotting."""
    return np.exp(-0.5 * x**2 - 0.5 * (y - x**2)**2)

# ============================================
# Real VI optimization with reparameterization trick
# ============================================

def vi_optimize(n_steps=1000, n_samples=32, lr=0.05):
    """
    Optimize mean-field Gaussian q(θ) = N(μ, diag(σ²)) to approximate banana posterior.
    Uses reparameterization trick for gradient estimation.
    """
    # Initialize variational parameters
    mu = np.array([0.0, 2.0])  # Start away from mode
    log_sigma = np.array([0.0, 0.0])  # log(sigma) for positivity

    # Storage for history
    history = {
        'elbo': [],
        'mu': [],
        'sigma': [],
    }

    for step in range(n_steps):
        sigma = np.exp(log_sigma)

        # Reparameterization: θ = μ + σ ⊙ ε, ε ~ N(0, I)
        eps = np.random.randn(n_samples, 2)
        theta = mu + sigma * eps  # (n_samples, 2)

        # ELBO = E_q[log p(θ)] - E_q[log q(θ)]
        # = E_q[log p(θ)] + H[q]
        # For Gaussian q: H[q] = 0.5 * d * (1 + log(2π)) + sum(log σ)

        log_p = banana_log_prob(theta)  # (n_samples,)
        entropy = 0.5 * 2 * (1 + np.log(2 * np.pi)) + np.sum(log_sigma)

        elbo = np.mean(log_p) + entropy

        # Gradients via reparameterization
        # ∂ELBO/∂μ = E_ε[∂log p / ∂θ]
        # ∂ELBO/∂σ = E_ε[∂log p / ∂θ ⊙ ε] + 1/σ

        # Gradient of log p(θ) w.r.t. θ
        x, y = theta[:, 0], theta[:, 1]
        dlog_p_dx = -x + 2 * x * (y - x**2)  # ∂/∂x [-x²/2 - (y-x²)²/2]
        dlog_p_dy = -(y - x**2)               # ∂/∂y [-x²/2 - (y-x²)²/2]
        grad_log_p = np.stack([dlog_p_dx, dlog_p_dy], axis=1)  # (n_samples, 2)

        # Gradients of ELBO
        grad_mu = np.mean(grad_log_p, axis=0)
        grad_log_sigma = np.mean(grad_log_p * eps, axis=0) * sigma + 1  # +1 from entropy

        # Gradient ascent
        mu = mu + lr * grad_mu
        log_sigma = log_sigma + lr * 0.5 * grad_log_sigma  # slower lr for variance

        # Clip log_sigma for stability
        log_sigma = np.clip(log_sigma, -3, 2)

        # Store
        history['elbo'].append(elbo)
        history['mu'].append(mu.copy())
        history['sigma'].append(np.exp(log_sigma).copy())

    return history

# Run optimization
print("Running VI optimization...")
history = vi_optimize(n_steps=800, n_samples=64, lr=0.03)

# ============================================
# Generate MCMC samples for comparison
# ============================================

def mh_sample(n_steps, step_size=0.5):
    """Metropolis-Hastings for banana distribution."""
    samples = []
    current = np.array([0.0, 1.0])

    for _ in range(n_steps):
        proposal = current + np.random.randn(2) * step_size
        log_ratio = banana_log_prob(proposal) - banana_log_prob(current)
        if np.log(np.random.rand()) < log_ratio:
            current = proposal
        samples.append(current.copy())

    return np.array(samples)

print("Running MCMC...")
np.random.seed(123)
mcmc_samples = mh_sample(2000, step_size=0.6)
mcmc_samples = mcmc_samples[200::4]  # Thin and remove burn-in

# ============================================
# Create figure
# ============================================

fig = plt.figure(figsize=(DOUBLE_COL, 4.5), constrained_layout=True)

# Layout: top row = MCMC + 3 VI snapshots, bottom = ELBO trace
gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.3)

ax_mcmc = fig.add_subplot(gs[0, 0])
ax_vi1 = fig.add_subplot(gs[0, 1])
ax_vi2 = fig.add_subplot(gs[0, 2])
ax_vi3 = fig.add_subplot(gs[0, 3])
ax_elbo = fig.add_subplot(gs[1, :])

# Grid for contours
x_grid = np.linspace(-2.5, 2.5, 150)
y_grid = np.linspace(-1.5, 4, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z_true = banana_density(X, Y)

# Snapshot steps
snapshots = [0, 100, 799]  # Start, middle, end
snapshot_labels = ['Step 0', 'Step 100', 'Step 800']

# Plot MCMC panel
ax_mcmc.contourf(X, Y, Z_true, levels=12, cmap='Blues', alpha=0.6)
ax_mcmc.contour(X, Y, Z_true, levels=5, colors='white', linewidths=0.4, alpha=0.5)
ax_mcmc.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], s=8, c=COLORS['orange'],
                alpha=0.6, edgecolors='white', linewidths=0.2)
ax_mcmc.set_xlabel(r'$\theta_1$')
ax_mcmc.set_ylabel(r'$\theta_2$')
ax_mcmc.set_title('MCMC samples')
ax_mcmc.set_xlim(-2.5, 2.5)
ax_mcmc.set_ylim(-1.5, 4)

# Legend for MCMC
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_mcmc = [
    Patch(facecolor=COLORS['blue'], alpha=0.5, label=r'$p(\theta \mid x)$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['orange'],
           markersize=5, label='Samples'),
]
ax_mcmc.legend(handles=legend_mcmc, loc='upper right', fontsize=6)

# Plot VI evolution panels
vi_axes = [ax_vi1, ax_vi2, ax_vi3]
for i, (ax, step, label) in enumerate(zip(vi_axes, snapshots, snapshot_labels)):
    # True posterior
    ax.contourf(X, Y, Z_true, levels=12, cmap='Blues', alpha=0.6)
    ax.contour(X, Y, Z_true, levels=5, colors='white', linewidths=0.4, alpha=0.5)

    # VI approximation at this step
    mu = history['mu'][step]
    sigma = history['sigma'][step]

    # Compute Gaussian q density
    rv = stats.multivariate_normal(mean=mu, cov=np.diag(sigma**2))
    Z_q = rv.pdf(np.stack([X, Y], axis=-1))

    # Plot q contours
    ax.contour(X, Y, Z_q, levels=4, colors=COLORS['orange'], linewidths=1.5, linestyles='--')

    # Mark the mean
    ax.scatter([mu[0]], [mu[1]], s=40, c=COLORS['red'], marker='x', linewidths=1.5, zorder=10)

    ax.set_xlabel(r'$\theta_1$')
    ax.set_title(label)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 4)
    ax.set_yticklabels([])

    # Add legend only to first VI panel
    if i == 0:
        legend_vi = [
            Line2D([0], [0], color=COLORS['orange'], linestyle='--', linewidth=1.5,
                   label=r'$q_\phi = \mathcal{N}(\mu, \sigma^2 I)$'),
        ]
        ax.legend(handles=legend_vi, loc='upper right', fontsize=6)

# Plot ELBO trace
steps = np.arange(len(history['elbo']))
elbo = np.array(history['elbo'])

ax_elbo.plot(steps, elbo, color=COLORS['blue'], linewidth=0.8, alpha=0.5)

# Smoothed ELBO
from scipy.ndimage import uniform_filter1d
elbo_smooth = uniform_filter1d(elbo, size=30)
ax_elbo.plot(steps, elbo_smooth, color=COLORS['blue'], linewidth=1.5, label='ELBO')

# Mark snapshot points
for step, label in zip(snapshots, snapshot_labels):
    ax_elbo.axvline(step, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.7)
    ax_elbo.scatter([step], [elbo_smooth[step]], s=50, c=COLORS['orange'], zorder=10,
                    edgecolors='white', linewidths=0.5)
    # Add label
    y_offset = 0.3 if step < 400 else 0.1
    ax_elbo.annotate(label, (step, elbo_smooth[step]),
                     xytext=(5, 10), textcoords='offset points', fontsize=7)

ax_elbo.set_xlabel('Optimization step')
ax_elbo.set_ylabel('ELBO')
ax_elbo.set_xlim(0, len(elbo))
ax_elbo.legend(loc='lower right')

fig.savefig('mcmc_vs_vi_overview.pdf')
print("Saved mcmc_vs_vi_overview.pdf")

# Print final VI parameters
print(f"\nFinal VI parameters:")
print(f"  μ = {history['mu'][-1]}")
print(f"  σ = {history['sigma'][-1]}")
print(f"  Final ELBO = {history['elbo'][-1]:.3f}")
