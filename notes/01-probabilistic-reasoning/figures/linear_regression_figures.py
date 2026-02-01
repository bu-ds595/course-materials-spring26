"""
Publication-quality figures for linear regression / Bayesian inference example.
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

# Set random seed for reproducibility
np.random.seed(42)

# True parameters
m_true = 2.5
c_true = 1.0
sigma = 1.5

# Generate data
N = 20
x = np.linspace(0, 5, N)
y_true = m_true * x + c_true
y = y_true + np.random.normal(0, sigma, N)

# MLE fit (OLS)
X = np.column_stack([x, np.ones(N)])
theta_mle = np.linalg.lstsq(X, y, rcond=None)[0]
m_mle, c_mle = theta_mle

# ============================================
# Figure 1: Data with fitted line
# ============================================
fig1, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.5), constrained_layout=True)

# Plot data points
ax1.scatter(x, y, s=30, color=COLORS['blue'], zorder=3, label='Data')

# Plot true line
x_line = np.linspace(-0.2, 5.2, 100)
ax1.plot(x_line, m_true * x_line + c_true, '--', color=COLORS['gray'],
         linewidth=1.2, label='True', zorder=2)

# Plot fitted line
ax1.plot(x_line, m_mle * x_line + c_mle, '-', color=COLORS['orange'],
         linewidth=1.5, label='Max. likelihood', zorder=2)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_xlim(-0.2, 5.2)
ax1.legend(loc='upper left', frameon=True)

fig1.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/figures/line_fit_data.pdf')
print("Saved line_fit_data.pdf")

# ============================================
# Figure 2: 2D Posterior contour plot
# ============================================

# Compute posterior (analytic for Gaussian likelihood + flat prior)
# Posterior covariance: (X^T X / sigma^2)^{-1}
XtX = X.T @ X
cov_post = sigma**2 * np.linalg.inv(XtX)
mean_post = theta_mle  # Posterior mean = MLE for flat prior

# Create grid for contour plot
m_range = np.linspace(1.5, 3.5, 200)
c_range = np.linspace(-1.0, 3.0, 200)
M, C = np.meshgrid(m_range, c_range)
pos = np.dstack((M, C))

# Multivariate normal PDF
rv = stats.multivariate_normal(mean_post, cov_post)
Z = rv.pdf(pos)

fig2, ax2 = plt.subplots(figsize=(SINGLE_COL, 2.8), constrained_layout=True)

# Filled contours
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
cf = ax2.contourf(M, C, Z / Z.max(), levels=np.linspace(0, 1, 15),
                   cmap='Blues', alpha=0.8)

# Contour lines at credible regions (approximate 1σ, 2σ)
# For 2D Gaussian: 1σ ≈ 39.3%, 2σ ≈ 86.5%
ax2.contour(M, C, Z / Z.max(), levels=[0.135, 0.607],
            colors=[COLORS['blue'], COLORS['blue']],
            linewidths=[1.0, 1.0], linestyles=['--', '-'])

# Mark true value
ax2.scatter([m_true], [c_true], marker='*', s=80, color=COLORS['orange'],
            zorder=5, label='True')

# Mark MLE
ax2.scatter([m_mle], [c_mle], marker='+', s=80, color=COLORS['red'],
            linewidths=2, zorder=5, label='Max. likelihood')

ax2.set_xlabel(r'Slope $m$')
ax2.set_ylabel(r'Intercept $c$')
ax2.legend(loc='upper right', frameon=True)

fig2.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/figures/line_posterior_2d.pdf')
print("Saved line_posterior_2d.pdf")

# ============================================
# Figure 3: Combined figure (data + posterior)
# ============================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), constrained_layout=True)

# Left panel: Data with posterior samples
ax3a.scatter(x, y, s=30, color=COLORS['blue'], zorder=3, label='Data')

# Draw posterior samples
np.random.seed(123)
for i in range(30):
    m_samp, c_samp = np.random.multivariate_normal(mean_post, cov_post)
    ax3a.plot(x_line, m_samp * x_line + c_samp, color=COLORS['teal'],
              alpha=0.15, linewidth=0.8, zorder=1)

# True line
ax3a.plot(x_line, m_true * x_line + c_true, '--', color='black',
         linewidth=1.2, label='True', zorder=2)

ax3a.set_xlabel(r'$x$')
ax3a.set_ylabel(r'$y$')
ax3a.set_xlim(-0.2, 5.2)
ax3a.set_title('(a) Data + posterior samples')
ax3a.legend(loc='upper left', frameon=True)

# Right panel: 2D posterior
cf = ax3b.contourf(M, C, Z / Z.max(), levels=np.linspace(0, 1, 15),
                    cmap='Blues', alpha=0.8)
ax3b.contour(M, C, Z / Z.max(), levels=[0.135, 0.607],
             colors=[COLORS['blue'], COLORS['blue']],
             linewidths=[1.0, 1.0], linestyles=['--', '-'])
ax3b.scatter([m_true], [c_true], marker='*', s=80, color=COLORS['orange'],
             zorder=5, label='True')
ax3b.scatter([m_mle], [c_mle], marker='+', s=80, color=COLORS['red'],
             linewidths=2, zorder=5, label='Max. likelihood')

ax3b.set_xlabel(r'Slope $m$')
ax3b.set_ylabel(r'Intercept $c$')
ax3b.set_title('(b) Joint posterior')
ax3b.legend(loc='upper right', frameon=True)

fig3.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/figures/line_regression_combined.pdf')
print("Saved line_regression_combined.pdf")

plt.show()
