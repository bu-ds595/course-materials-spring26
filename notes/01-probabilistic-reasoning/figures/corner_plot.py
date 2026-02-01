"""
Corner plot for line-fitting posterior showing 68% and 95% credible regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.patches import Patch

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

# Set random seed for reproducibility
np.random.seed(21)

# True parameters
m_true = 2.5
c_true = 1.0
sigma = 1.5

# Generate data (t = input, x = response/data)
N = 20
t = np.linspace(0, 5, N)
x_true = m_true * t + c_true
x_data = x_true + np.random.normal(0, sigma, N)

# MLE fit (OLS)
T = np.column_stack([t, np.ones(N)])
theta_mle = np.linalg.lstsq(T, x_data, rcond=None)[0]
m_mle, c_mle = theta_mle

# Compute posterior (analytic for Gaussian likelihood + flat prior)
TtT = T.T @ T
cov_post = sigma**2 * np.linalg.inv(TtT)
mean_post = theta_mle

# Create grid for contour plot
m_range = np.linspace(1.5, 3.5, 200)
c_range = np.linspace(-1.0, 3.0, 200)
M, C = np.meshgrid(m_range, c_range)
pos = np.dstack((M, C))

# Multivariate normal PDF
rv = stats.multivariate_normal(mean_post, cov_post)
Z = rv.pdf(pos)
Z_norm = Z / Z.max()

# Compute credible region levels
# For 2D Gaussian, the fraction of probability mass inside a contour at level L
# is 1 - L. So for 68% we need L = 0.32, for 95% we need L = 0.05
# But we're working with normalized Z, so we need chi2 levels
# Delta chi2 = 2.30 for 68.3% (1 sigma), 6.17 for 95.4% (2 sigma) in 2D
level_68 = np.exp(-2.30 / 2)  # ~0.317
level_95 = np.exp(-6.17 / 2)  # ~0.046

# ============================================
# Corner plot
# ============================================
fig = plt.figure(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.4))

# Create grid: 2x2 with marginals on diagonal
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                       wspace=0.08, hspace=0.08)

# Top-left: marginal for m
ax_m = fig.add_subplot(gs[0, 0])
marginal_m = np.exp(-0.5 * (m_range - mean_post[0])**2 / cov_post[0, 0])
ax_m.fill_between(m_range, marginal_m, alpha=0.3, color=COLORS['teal'])
ax_m.plot(m_range, marginal_m, color=COLORS['teal'], linewidth=1.5)

# Shade 68% credible interval
std_m = np.sqrt(cov_post[0, 0])
mask_68 = (m_range >= mean_post[0] - std_m) & (m_range <= mean_post[0] + std_m)
ax_m.fill_between(m_range, marginal_m, where=mask_68, alpha=0.4, color=COLORS['teal'])

ax_m.axvline(m_true, color=COLORS['orange'], linestyle='--', linewidth=1, alpha=0.8)
ax_m.set_xlim(m_range[0], m_range[-1])
ax_m.set_ylim(0, 1.15)
ax_m.set_ylabel('Density')
ax_m.tick_params(labelbottom=False)
ax_m.set_title(r'$p(m \mid \mathrm{data})$', fontsize=9)

# Bottom-right: marginal for c
ax_c = fig.add_subplot(gs[1, 1])
marginal_c = np.exp(-0.5 * (c_range - mean_post[1])**2 / cov_post[1, 1])
ax_c.fill_betweenx(c_range, marginal_c, alpha=0.3, color=COLORS['teal'])
ax_c.plot(marginal_c, c_range, color=COLORS['teal'], linewidth=1.5)

# Shade 68% credible interval
std_c = np.sqrt(cov_post[1, 1])
mask_68_c = (c_range >= mean_post[1] - std_c) & (c_range <= mean_post[1] + std_c)
ax_c.fill_betweenx(c_range, marginal_c, where=mask_68_c, alpha=0.4, color=COLORS['teal'])

ax_c.axhline(c_true, color=COLORS['orange'], linestyle='--', linewidth=1, alpha=0.8)
ax_c.set_ylim(c_range[0], c_range[-1])
ax_c.set_xlim(0, 1.15)
ax_c.set_xlabel('Density')
ax_c.tick_params(labelleft=False)
ax_c.set_title(r'$p(c \mid \mathrm{data})$', fontsize=9)

# Bottom-left: joint posterior
ax_joint = fig.add_subplot(gs[1, 0])

# Filled contours for 68% and 95% regions
ax_joint.contourf(M, C, Z_norm, levels=[level_95, 1.0], colors=[COLORS['teal']], alpha=0.2)
ax_joint.contourf(M, C, Z_norm, levels=[level_68, 1.0], colors=[COLORS['teal']], alpha=0.3)

# Contour lines
ax_joint.contour(M, C, Z_norm, levels=[level_95, level_68],
                 colors=[COLORS['teal']], linewidths=[0.8, 1.2])

# Mark true value
ax_joint.scatter([m_true], [c_true], marker='*', s=60, color=COLORS['orange'],
                 zorder=5, edgecolors='white', linewidths=0.5)

ax_joint.set_xlabel(r'Slope $m$')
ax_joint.set_ylabel(r'Intercept $c$')
ax_joint.set_xlim(m_range[0], m_range[-1])
ax_joint.set_ylim(c_range[0], c_range[-1])

# Top-right: legend
ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.axis('off')

legend_elements = [
    Patch(facecolor=COLORS['teal'], alpha=0.5, edgecolor=COLORS['teal'],
          linewidth=1.2, label='68% credible region'),
    Patch(facecolor=COLORS['teal'], alpha=0.2, edgecolor=COLORS['teal'],
          linewidth=0.8, label='95% credible region'),
    plt.Line2D([0], [0], color=COLORS['orange'], linestyle='--',
               linewidth=1, marker='*', markersize=8, label='True value'),
]
ax_legend.legend(handles=legend_elements, loc='center', frameon=True, fontsize=7)

fig.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/corner_plot.pdf')
print("Saved corner_plot.pdf")

plt.show()
