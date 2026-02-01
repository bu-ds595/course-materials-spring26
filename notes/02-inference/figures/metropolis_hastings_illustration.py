"""
Metropolis-Hastings algorithm illustration.

Shows the key components of MH:
- Prior distribution (left)
- Chain evolution with accepted/rejected proposals (center)
- Resulting posterior samples forming a histogram (right)
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
# Define target posterior (unnormalized)
# ============================================

# Target: a skewed distribution (mixture of Gaussians)
def log_posterior(theta):
    """Log of unnormalized posterior density."""
    # Mixture of two Gaussians centered at 0.3 and 0.7
    p1 = 0.6 * stats.norm.pdf(theta, 0.35, 0.12)
    p2 = 0.4 * stats.norm.pdf(theta, 0.72, 0.08)
    return np.log(p1 + p2 + 1e-10)

def posterior(theta):
    """Unnormalized posterior density."""
    p1 = 0.6 * stats.norm.pdf(theta, 0.35, 0.12)
    p2 = 0.4 * stats.norm.pdf(theta, 0.72, 0.08)
    return p1 + p2

# Prior: uniform on [0, 1]
prior_lb = 0.0
prior_ub = 1.0

# ============================================
# Run Metropolis-Hastings
# ============================================

def metropolis_hastings(n_steps, step_size=0.1, start=0.8):
    """Run MH with Gaussian proposals, tracking accepts/rejects."""
    samples = [start]
    proposals = []  # (from_theta, to_theta, accepted)
    current = start

    for _ in range(n_steps):
        # Propose
        proposal = current + np.random.randn() * step_size

        # Reflect at boundaries for uniform prior
        if proposal < prior_lb:
            proposal = prior_lb + (prior_lb - proposal)
        if proposal > prior_ub:
            proposal = prior_ub - (proposal - prior_ub)
        proposal = np.clip(proposal, prior_lb, prior_ub)

        # Accept/reject
        log_ratio = log_posterior(proposal) - log_posterior(current)
        accepted = np.log(np.random.rand()) < log_ratio

        proposals.append((current, proposal, accepted))

        if accepted:
            current = proposal
        samples.append(current)

    return np.array(samples), proposals

# Run chain
n_steps = 35
samples, proposals = metropolis_hastings(n_steps, step_size=0.12, start=0.85)

# ============================================
# Create figure
# ============================================

fig = plt.figure(figsize=(DOUBLE_COL, 2.8), constrained_layout=True)

# Create grid: prior (narrow) | chain (wide) | posterior (narrow)
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 3, 1.2], wspace=0.1)

ax_prior = fig.add_subplot(gs[0])
ax_chain = fig.add_subplot(gs[1])
ax_post = fig.add_subplot(gs[2])

# ============================================
# Left panel: Prior distribution
# ============================================

theta_grid = np.linspace(-0.05, 1.05, 200)
prior_density = np.where((theta_grid >= prior_lb) & (theta_grid <= prior_ub), 1.0, 0.0)

ax_prior.fill_betweenx(theta_grid, 0, prior_density, alpha=0.3, color=COLORS['blue'])
ax_prior.plot(prior_density, theta_grid, color=COLORS['blue'], linewidth=1.5)

ax_prior.set_ylim(prior_lb - 0.05, prior_ub + 0.05)
ax_prior.set_xlim(-0.1, 1.5)
ax_prior.set_ylabel(r'$\theta$')
ax_prior.set_xlabel(r'$p(\theta)$')
ax_prior.set_title('Prior')
ax_prior.set_xticks([0, 1])

# ============================================
# Middle panel: Chain evolution
# ============================================

iterations = np.arange(len(samples))

# Draw accepted transitions (solid) and rejected proposals (dashed)
for i, (from_theta, to_theta, accepted) in enumerate(proposals):
    if accepted:
        # Solid line for accepted move
        ax_chain.plot([i, i+1], [from_theta, to_theta],
                     color=COLORS['blue'], linewidth=1.2, zorder=2)
    else:
        # Dashed line showing rejected proposal
        ax_chain.plot([i, i+0.5], [from_theta, to_theta],
                     color=COLORS['red'], linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)
        # Solid line staying in place
        ax_chain.plot([i, i+1], [from_theta, from_theta],
                     color=COLORS['blue'], linewidth=1.2, zorder=2)

# Draw sample points
ax_chain.scatter(iterations, samples, s=25, c=COLORS['orange'],
                zorder=5, edgecolors='white', linewidths=0.4)

# Mark initial sample
ax_chain.scatter([0], [samples[0]], s=60, c=COLORS['orange'],
                marker='s', zorder=6, edgecolors='white', linewidths=0.5)
ax_chain.annotate(r'$\theta^{(0)}$', xy=(0, samples[0]), xytext=(-0.5, samples[0] + 0.08),
                 fontsize=8, ha='center')

ax_chain.set_xlim(-1, n_steps + 1)
ax_chain.set_ylim(prior_lb - 0.05, prior_ub + 0.05)
ax_chain.set_xlabel('Iteration $t$')
ax_chain.set_title('Markov chain')
ax_chain.set_yticks([])

# Add legend for chain panel
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
chain_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['orange'],
           markersize=6, label='Accepted sample'),
    Line2D([0], [0], color=COLORS['blue'], linewidth=1.2, label='Accepted transition'),
    Line2D([0], [0], color=COLORS['red'], linewidth=0.8, linestyle='--',
           alpha=0.6, label='Rejected proposal'),
]
ax_chain.legend(handles=chain_legend, loc='lower left', fontsize=6)

# ============================================
# Right panel: Posterior distribution
# ============================================

# Compute true posterior for comparison
theta_fine = np.linspace(prior_lb, prior_ub, 200)
post_density = posterior(theta_fine)
post_density = post_density / np.trapz(post_density, theta_fine)  # Normalize

# Plot true posterior
ax_post.fill_betweenx(theta_fine, 0, post_density, alpha=0.3, color=COLORS['blue'])
ax_post.plot(post_density, theta_fine, color=COLORS['blue'], linewidth=1.5, label=r'$p(\theta \mid x)$')

# Plot histogram of samples (rotated)
# Use only samples after brief burn-in
burn_in = 5
hist_samples = samples[burn_in:]
bins = np.linspace(prior_lb, prior_ub, 12)
counts, bin_edges = np.histogram(hist_samples, bins=bins, density=True)

# Draw histogram as horizontal bars
bar_height = bin_edges[1] - bin_edges[0]
bar_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax_post.barh(bar_centers, counts, height=bar_height * 0.9,
            alpha=0.5, color=COLORS['orange'], edgecolor=COLORS['orange'], linewidth=0.5)

ax_post.set_ylim(prior_lb - 0.05, prior_ub + 0.05)
ax_post.set_xlim(0, max(post_density) * 1.3)
ax_post.set_xlabel(r'$p(\theta \mid x)$')
ax_post.set_title('Posterior')
ax_post.set_yticks([])
ax_post.set_xticks([])

# Add legend
legend_elements = [
    Patch(facecolor=COLORS['blue'], alpha=0.3, edgecolor=COLORS['blue'], label='True posterior'),
    Patch(facecolor=COLORS['orange'], alpha=0.5, edgecolor=COLORS['orange'], label='MCMC histogram'),
]
ax_post.legend(handles=legend_elements, loc='upper right', fontsize=6)

# Save
fig.savefig('metropolis_hastings_illustration.pdf')
print("Saved metropolis_hastings_illustration.pdf")
