"""
Simple MCMC vs VI intro figure: just samples (left) and q_φ approximation (right).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
# Banana posterior
# ============================================

def banana_log_prob(theta):
    x, y = theta[..., 0], theta[..., 1]
    return -0.5 * x**2 - 0.5 * (y - x**2)**2

def banana_density(x, y):
    return np.exp(-0.5 * x**2 - 0.5 * (y - x**2)**2)

# ============================================
# Generate MCMC samples
# ============================================

def mh_sample(n_steps, step_size=0.5):
    samples = []
    current = np.array([0.0, 1.0])
    for _ in range(n_steps):
        proposal = current + np.random.randn(2) * step_size
        log_ratio = banana_log_prob(proposal) - banana_log_prob(current)
        if np.log(np.random.rand()) < log_ratio:
            current = proposal
        samples.append(current.copy())
    return np.array(samples)

np.random.seed(123)
mcmc_samples = mh_sample(2000, step_size=0.6)
mcmc_samples = mcmc_samples[200::4]  # Thin and remove burn-in

# ============================================
# VI approximation (converged)
# ============================================

# These are roughly the converged values from real optimization
vi_mean = np.array([0.0, 0.4])
vi_std = np.array([0.65, 1.0])

# ============================================
# Create figure
# ============================================

fig, (ax_mcmc, ax_vi) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), constrained_layout=True)

# Grid for contours
x_grid = np.linspace(-2.5, 2.5, 150)
y_grid = np.linspace(-1.5, 4, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z_true = banana_density(X, Y)

# VI Gaussian density
rv = stats.multivariate_normal(mean=vi_mean, cov=np.diag(vi_std**2))
Z_vi = rv.pdf(np.stack([X, Y], axis=-1))

# Left panel: MCMC samples
ax_mcmc.contourf(X, Y, Z_true, levels=12, cmap='Blues', alpha=0.6)
ax_mcmc.contour(X, Y, Z_true, levels=5, colors='white', linewidths=0.4, alpha=0.5)
ax_mcmc.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], s=10, c=COLORS['orange'],
                alpha=0.7, edgecolors='white', linewidths=0.2)

ax_mcmc.set_xlabel(r'$\theta_1$')
ax_mcmc.set_ylabel(r'$\theta_2$')
ax_mcmc.set_title('MCMC: samples from posterior')
ax_mcmc.set_xlim(-2.5, 2.5)
ax_mcmc.set_ylim(-1.5, 4)

# Legend for left panel
legend_elements = [
    Patch(facecolor=COLORS['blue'], alpha=0.5, label=r'True posterior $p(\theta \mid x)$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['orange'],
           markersize=6, label='MCMC samples'),
]
ax_mcmc.legend(handles=legend_elements, loc='upper right', fontsize=7)

# Right panel: VI approximation
ax_vi.contourf(X, Y, Z_true, levels=12, cmap='Blues', alpha=0.6)
ax_vi.contour(X, Y, Z_true, levels=5, colors='white', linewidths=0.4, alpha=0.5)
ax_vi.contour(X, Y, Z_vi, levels=4, colors=COLORS['orange'], linewidths=1.5, linestyles='--')

ax_vi.set_xlabel(r'$\theta_1$')
ax_vi.set_title(r'VI: approximate $q_\phi(\theta) \approx p(\theta \mid x)$')
ax_vi.set_xlim(-2.5, 2.5)
ax_vi.set_ylim(-1.5, 4)

# Legend for right panel
legend_elements_vi = [
    Patch(facecolor=COLORS['blue'], alpha=0.5, label=r'True posterior $p(\theta \mid x)$'),
    Line2D([0], [0], color=COLORS['orange'], linestyle='--', linewidth=1.5,
           label=r'$q_\phi = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$'),
]
ax_vi.legend(handles=legend_elements_vi, loc='upper right', fontsize=7)

fig.savefig('mcmc_vs_vi_intro.pdf')
print("Saved mcmc_vs_vi_intro.pdf")
