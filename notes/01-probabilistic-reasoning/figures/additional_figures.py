"""
Additional publication-quality figures for Chapter 1: Probabilistic Reasoning
1. Marginalization visualization
2. Prior → Posterior with increasing data
3. Generative vs Discriminative models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
# Figure 1: Marginalization
# ============================================
# Show a 2D joint distribution and its marginals

fig1 = plt.figure(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3), constrained_layout=True)

# Create grid for main plot and marginals
gs = fig1.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                        wspace=0.05, hspace=0.05)

ax_joint = fig1.add_subplot(gs[1, 0])
ax_marg_x = fig1.add_subplot(gs[0, 0], sharex=ax_joint)
ax_marg_y = fig1.add_subplot(gs[1, 1], sharey=ax_joint)

# Create a correlated 2D Gaussian
mean = [0, 0]
cov = [[1.0, 0.6], [0.6, 0.8]]
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_range, y_range)
pos = np.dstack((X, Y))
rv = stats.multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Joint distribution (center)
levels = np.linspace(0.01, Z.max(), 12)
cf = ax_joint.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.8)
ax_joint.contour(X, Y, Z, levels=levels[::2], colors=COLORS['blue'], linewidths=0.5, alpha=0.7)
ax_joint.set_xlabel(r'$x$')
ax_joint.set_ylabel(r'$y$')
ax_joint.set_xlim(-3, 3)
ax_joint.set_ylim(-3, 3)

# Marginal p(x) - integrate out y
marginal_x = Z.sum(axis=0) * (y_range[1] - y_range[0])
marginal_x = marginal_x / marginal_x.max() * 0.9  # Normalize for display
ax_marg_x.fill_between(x_range, marginal_x, alpha=0.4, color=COLORS['orange'])
ax_marg_x.plot(x_range, marginal_x, color=COLORS['orange'], linewidth=1.5)
ax_marg_x.set_ylabel(r'$p(x)$', fontsize=8)
ax_marg_x.set_ylim(0, 1.1)
ax_marg_x.tick_params(labelbottom=False)
ax_marg_x.set_title(r'$p(x) = \int p(x,y)\, dy$', fontsize=8, pad=3)

# Marginal p(y) - integrate out x
marginal_y = Z.sum(axis=1) * (x_range[1] - x_range[0])
marginal_y = marginal_y / marginal_y.max() * 0.9
ax_marg_y.fill_betweenx(y_range, marginal_y, alpha=0.4, color=COLORS['teal'])
ax_marg_y.plot(marginal_y, y_range, color=COLORS['teal'], linewidth=1.5)
ax_marg_y.set_xlabel(r'$p(y)$', fontsize=8)
ax_marg_y.set_xlim(0, 1.1)
ax_marg_y.tick_params(labelleft=False)
ax_marg_y.set_title(r'$p(y) = \int p(x,y)\, dx$', fontsize=8, pad=3)

# Add annotation for joint
ax_joint.text(0, 0, r'$p(x,y)$', fontsize=10, ha='center', va='center',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig1.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/marginalization.pdf')
print("Saved marginalization.pdf")


# ============================================
# Figure 2: Prior → Posterior with increasing data
# ============================================

fig2, axes2 = plt.subplots(1, 4, figsize=(DOUBLE_COL, 1.8), constrained_layout=True)

theta_range = np.linspace(-1, 5, 500)
theta_true = 2.0
sigma = 1.0

# Prior: broad Gaussian centered at 1
prior_mean, prior_std = 1.5, 1.5
prior = stats.norm.pdf(theta_range, prior_mean, prior_std)

# Generate data
np.random.seed(42)
all_data = np.random.normal(theta_true, sigma, 50)

n_values = [0, 1, 5, 20]
titles = [r'$p(\theta)$', r'$p(\theta \mid x_1)$', r'$p(\theta \mid x_{1:5})$', r'$p(\theta \mid x_{1:20})$']

for i, (n, title) in enumerate(zip(n_values, titles)):
    ax = axes2[i]

    if n == 0:
        # Just show prior
        ax.fill_between(theta_range, prior / prior.max(), alpha=0.4, color=COLORS['blue'])
        ax.plot(theta_range, prior / prior.max(), color=COLORS['blue'], linewidth=1.5)
    else:
        # Compute posterior (conjugate Gaussian)
        data = all_data[:n]
        data_mean = np.mean(data)

        # Posterior for Gaussian likelihood with known variance
        # Prior: N(prior_mean, prior_std^2)
        # Likelihood: product of N(theta, sigma^2)
        prior_precision = 1 / prior_std**2
        likelihood_precision = n / sigma**2
        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1 / posterior_precision
        posterior_mean = posterior_var * (prior_precision * prior_mean + likelihood_precision * data_mean)
        posterior_std = np.sqrt(posterior_var)

        posterior = stats.norm.pdf(theta_range, posterior_mean, posterior_std)

        # Show prior faintly
        ax.plot(theta_range, prior / prior.max(), '--', color=COLORS['blue'],
                linewidth=0.8, alpha=0.4)

        # Show posterior
        ax.fill_between(theta_range, posterior / posterior.max(), alpha=0.4, color=COLORS['teal'])
        ax.plot(theta_range, posterior / posterior.max(), color=COLORS['teal'], linewidth=1.5)

    # Mark true value
    ax.axvline(theta_true, color=COLORS['orange'], linestyle=':', linewidth=1, alpha=0.8)

    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r'$\theta$')
    if i == 0:
        ax.set_ylabel('Density')
    ax.set_title(title, fontsize=9)
    ax.set_yticks([])

# Add legend to first panel
legend_elements = [
    Line2D([0], [0], color=COLORS['blue'], linestyle='--', linewidth=1, alpha=0.6, label='Prior'),
    Line2D([0], [0], color=COLORS['teal'], linewidth=1.5, label='Posterior'),
    Line2D([0], [0], color=COLORS['orange'], linestyle=':', linewidth=1, label='True')
]
axes2[0].legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=6)

fig2.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/posterior_evolution.pdf')
print("Saved posterior_evolution.pdf")


# ============================================
# Figure 3: Generative vs Discriminative
# ============================================

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0), constrained_layout=True)

# Helper function to draw a box with text
def draw_box(ax, center, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((center[0] - width/2, center[1] - height/2),
                          width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5,
                          alpha=0.3)
    ax.add_patch(box)
    ax.text(center[0], center[1], text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, start, end, color='black', label=None, label_pos='above'):
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=2)
    ax.add_patch(arrow)
    if label:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        offset = 0.25 if label_pos == 'above' else -0.25
        ax.text(mid[0], mid[1] + offset, label, ha='center', va='center',
                fontsize=9, color=color)

# Panel (a): Discriminative
ax3a.set_xlim(-0.5, 4.5)
ax3a.set_ylim(-0.3, 2.0)
ax3a.set_aspect('equal')
ax3a.axis('off')
ax3a.set_title('(a) Discriminative', fontsize=10, pad=8)

# x -> theta
draw_box(ax3a, (0.8, 0.8), 1.0, 0.7, r'$x$', COLORS['blue'])
draw_box(ax3a, (3.2, 0.8), 1.0, 0.7, r'$\theta$', COLORS['orange'])
draw_arrow(ax3a, (1.4, 0.8), (2.6, 0.8), COLORS['teal'], r'$p(\theta \mid x)$')

# Label
ax3a.text(2, 1.7, 'Learn posterior directly', ha='center', fontsize=8,
          style='italic', color='gray')

# Panel (b): Generative
ax3b.set_xlim(-0.5, 4.5)
ax3b.set_ylim(-0.3, 2.0)
ax3b.set_aspect('equal')
ax3b.axis('off')
ax3b.set_title('(b) Generative', fontsize=10, pad=8)

# Show data distribution box
draw_box(ax3b, (2.0, 0.8), 2.4, 0.7, r'$p(x), \; p(x,\theta), \; p(x \mid \theta)$', COLORS['teal'], fontsize=9)

# Label
ax3b.text(2, 1.7, 'Model the data distribution', ha='center', fontsize=8,
          style='italic', color='gray')

fig3.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/generative_discriminative.pdf')
print("Saved generative_discriminative.pdf")


print("\nAll additional figures generated successfully!")
