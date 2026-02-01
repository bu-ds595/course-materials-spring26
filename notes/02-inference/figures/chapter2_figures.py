"""
Publication-quality figures for Chapter 2: Inference (MCMC and VI).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

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
# Figure 0: MCMC vs VI overview (chapter opener)
# ============================================

fig0, (ax0a, ax0b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), constrained_layout=True)

# Banana-shaped posterior
def banana_posterior(x, y):
    """Curved posterior: p(x,y) ~ exp(-x^2/2 - (y - x^2)^2/2)"""
    return np.exp(-0.5 * x**2 - 0.5 * (y - x**2)**2)

x_grid = np.linspace(-2.5, 2.5, 200)
y_grid = np.linspace(-1, 4, 200)
X0, Y0 = np.meshgrid(x_grid, y_grid)
Z0 = banana_posterior(X0, Y0)

# Generate MCMC samples via Metropolis-Hastings
def banana_log_prob(theta):
    x, y = theta
    return -0.5 * x**2 - 0.5 * (y - x**2)**2

def mh_banana(start, n_steps, step_size=0.5):
    samples = [start]
    current = start
    for _ in range(n_steps):
        proposal = current + np.random.randn(2) * step_size
        log_ratio = banana_log_prob(proposal) - banana_log_prob(current)
        if np.log(np.random.rand()) < log_ratio:
            current = proposal
        samples.append(current.copy())
    return np.array(samples)

np.random.seed(42)
mcmc_samples = mh_banana(np.array([0.0, 1.0]), 800, step_size=0.6)
# Thin and remove burn-in
mcmc_samples = mcmc_samples[100::3]

# VI approximation: mean-field Gaussian that captures the mode
# After optimization, VI would find something like this
vi_mean = np.array([0.0, 1.0])
vi_std = np.array([0.9, 0.8])

def vi_approx(x, y):
    return stats.multivariate_normal.pdf(
        np.stack([x, y], axis=-1), mean=vi_mean, cov=np.diag(vi_std**2)
    )

Z0_vi = vi_approx(X0, Y0)

# Left panel: MCMC samples on true posterior
ax0a.contourf(X0, Y0, Z0, levels=15, cmap='Blues', alpha=0.6)
ax0a.contour(X0, Y0, Z0, levels=6, colors='white', linewidths=0.5, alpha=0.5)
ax0a.scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], s=12, c=COLORS['orange'],
             alpha=0.7, zorder=5, edgecolors='white', linewidths=0.3, label='MCMC samples')
ax0a.set_xlabel(r'$\theta_1$')
ax0a.set_ylabel(r'$\theta_2$')
ax0a.set_title('MCMC: samples from posterior', fontsize=9)
ax0a.set_xlim(-2.5, 2.5)
ax0a.set_ylim(-1, 4)
ax0a.legend(fontsize=7, loc='upper right')

# Right panel: VI approximation on true posterior
ax0b.contourf(X0, Y0, Z0, levels=15, cmap='Blues', alpha=0.6)
ax0b.contour(X0, Y0, Z0, levels=6, colors='white', linewidths=0.5, alpha=0.5)
# Overlay VI contours
ax0b.contour(X0, Y0, Z0_vi, levels=4, colors=COLORS['orange'], linewidths=1.5, linestyles='--')
ax0b.set_xlabel(r'$\theta_1$')
ax0b.set_title(r'VI: approximate $q_\phi(\theta) \approx p(\theta \mid x)$', fontsize=9)
ax0b.set_xlim(-2.5, 2.5)
ax0b.set_ylim(-1, 4)

# Add legend patch for VI
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
vi_line = Line2D([0], [0], color=COLORS['orange'], linestyle='--', linewidth=1.5, label=r'$q_\phi(\theta)$')
ax0b.legend(handles=[vi_line], fontsize=7, loc='upper right')

fig0.savefig('mcmc_vs_vi_overview.pdf')
print("Saved mcmc_vs_vi_overview.pdf")


# ============================================
# Figure 1: MCMC on a 2D landscape
# ============================================

def target_density(x, y):
    """Bimodal target distribution."""
    return 0.6 * stats.multivariate_normal.pdf(
        np.stack([x, y], axis=-1), mean=[-1, 0], cov=[[0.5, 0.3], [0.3, 0.5]]
    ) + 0.4 * stats.multivariate_normal.pdf(
        np.stack([x, y], axis=-1), mean=[1.5, 1], cov=[[0.3, -0.1], [-0.1, 0.4]]
    )

fig1, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL), constrained_layout=True)

# Create density grid
x = np.linspace(-3, 4, 200)
y = np.linspace(-2, 3, 200)
X, Y = np.meshgrid(x, y)
Z = target_density(X, Y)

# Plot contours
ax1.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.7)
ax1.contour(X, Y, Z, levels=6, colors='white', linewidths=0.5, alpha=0.5)

# Simulate a short MH chain
def mh_sample(start, n_steps, step_size=0.3):
    samples = [start]
    current = start
    for _ in range(n_steps):
        proposal = current + np.random.randn(2) * step_size
        ratio = target_density(proposal[0], proposal[1]) / target_density(current[0], current[1])
        if np.random.rand() < ratio:
            current = proposal
        samples.append(current.copy())
    return np.array(samples)

chain = mh_sample(np.array([-2.5, -1.5]), 60, step_size=0.4)

# Plot chain
ax1.plot(chain[:, 0], chain[:, 1], '-', color=COLORS['orange'], alpha=0.7, linewidth=1)
ax1.scatter(chain[:, 0], chain[:, 1], s=15, c=COLORS['orange'], zorder=5, edgecolors='white', linewidths=0.3)
ax1.scatter([chain[0, 0]], [chain[0, 1]], s=50, c=COLORS['red'], marker='s', zorder=6, label='Start')

ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.set_xlim(-3, 4)
ax1.set_ylim(-2, 3)
ax1.set_title('Metropolis-Hastings random walk', fontsize=9)

fig1.savefig('mcmc_landscape.pdf')
print("Saved mcmc_landscape.pdf")


# ============================================
# Figure 2: Step size comparison
# ============================================

fig2, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), constrained_layout=True)

# Single mode for clarity
def simple_target(x, y):
    return stats.multivariate_normal.pdf(
        np.stack([x, y], axis=-1), mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]
    )

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = simple_target(X, Y)

step_sizes = [0.05, 0.5, 3.0]
titles = ['Too small: slow exploration', 'Good: efficient mixing', 'Too large: many rejections']

for ax, step_size, title in zip(axes, step_sizes, titles):
    ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6)
    ax.contour(X, Y, Z, levels=4, colors='white', linewidths=0.5, alpha=0.5)

    # Run chain
    chain = mh_sample(np.array([-2, -2]), 80, step_size=step_size)
    ax.plot(chain[:, 0], chain[:, 1], '-', color=COLORS['orange'], alpha=0.6, linewidth=0.8)
    ax.scatter(chain[:, 0], chain[:, 1], s=8, c=COLORS['orange'], zorder=5, edgecolors='white', linewidths=0.2)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r'$\theta_1$')
    if ax == axes[0]:
        ax.set_ylabel(r'$\theta_2$')

fig2.savefig('step_size.pdf')
print("Saved step_size.pdf")


# ============================================
# Figure 3: Typical set in high dimensions
# ============================================

fig3, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), constrained_layout=True)

dims = [2, 10, 50]

for ax, d in zip(axes, dims):
    # For a d-dimensional standard Gaussian, radius^2 ~ chi^2(d)
    # Expected radius: sqrt(d), std: 1/sqrt(2)
    r = np.linspace(0, np.sqrt(d) + 4, 500)

    # Density at radius r in d dimensions: proportional to r^(d-1) * exp(-r^2/2)
    # This is the chi distribution with d degrees of freedom
    chi_pdf = stats.chi.pdf(r, d)

    ax.fill_between(r, chi_pdf, alpha=0.3, color=COLORS['blue'])
    ax.plot(r, chi_pdf, color=COLORS['blue'], linewidth=1.5)

    # Mark the mode (typical set location)
    mode = np.sqrt(d - 1) if d > 1 else 0
    ax.axvline(mode, color=COLORS['orange'], linestyle='--', linewidth=1, label=f'Mode: $\\sqrt{{{d-1}}}$')

    ax.set_xlabel(r'Distance from origin $\|\theta\|$')
    if ax == axes[0]:
        ax.set_ylabel('Probability density')
    ax.set_title(f'd = {d}', fontsize=9)
    ax.set_xlim(0, r[-1])
    ax.set_ylim(0, None)

axes[0].legend(fontsize=7, loc='upper right')

fig3.savefig('typical_set.pdf')
print("Saved typical_set.pdf")


# ============================================
# Figure 3b: Gradient information illustration
# ============================================

fig3b, (ax3b_left, ax3b_right) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), constrained_layout=True)

# Correlated Gaussian for both panels
cov_grad = np.array([[1, 0.7], [0.7, 1]])
mean_grad = np.array([0, 0])
inv_cov_grad = np.linalg.inv(cov_grad)

x_grad = np.linspace(-3, 3, 100)
y_grad = np.linspace(-3, 3, 100)
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)
Z_grad = np.exp(-0.5 * (inv_cov_grad[0,0]*X_grad**2 + 2*inv_cov_grad[0,1]*X_grad*Y_grad + inv_cov_grad[1,1]*Y_grad**2))

# Left panel: Random walk proposals (isotropic, ignoring structure)
ax3b_left.contourf(X_grad, Y_grad, Z_grad, levels=12, cmap='Blues', alpha=0.6)
ax3b_left.contour(X_grad, Y_grad, Z_grad, levels=6, colors='white', linewidths=0.5, alpha=0.6)

# Show a point and random proposal directions
point = np.array([1.2, 0.8])
ax3b_left.scatter([point[0]], [point[1]], s=60, c=COLORS['orange'], zorder=10, edgecolors='white', linewidths=1)

# Random directions (isotropic)
np.random.seed(42)
n_arrows = 8
angles = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
arrow_len = 0.7
for angle in angles:
    dx, dy = arrow_len * np.cos(angle), arrow_len * np.sin(angle)
    ax3b_left.annotate('', xy=(point[0]+dx, point[1]+dy), xytext=(point[0], point[1]),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5, mutation_scale=12))

ax3b_left.set_xlabel(r'$\theta_1$')
ax3b_left.set_ylabel(r'$\theta_2$')
ax3b_left.set_title('Random walk: proposes equally in all directions', fontsize=8)
ax3b_left.set_xlim(-3, 3)
ax3b_left.set_ylim(-3, 3)
ax3b_left.set_aspect('equal')

# Right panel: HMC uses gradient information
ax3b_right.contourf(X_grad, Y_grad, Z_grad, levels=12, cmap='Blues', alpha=0.6)
ax3b_right.contour(X_grad, Y_grad, Z_grad, levels=6, colors='white', linewidths=0.5, alpha=0.6)

# Same point
ax3b_right.scatter([point[0]], [point[1]], s=60, c=COLORS['orange'], zorder=10, edgecolors='white', linewidths=1)

# Gradient direction (points toward mode)
grad = -inv_cov_grad @ point  # gradient of log p
grad_norm = grad / np.linalg.norm(grad)
grad_len = 0.9

# Draw gradient arrow (red, toward higher probability)
ax3b_right.annotate('', xy=(point[0]+grad_len*grad_norm[0], point[1]+grad_len*grad_norm[1]),
                    xytext=(point[0], point[1]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2, mutation_scale=15))
ax3b_right.text(point[0]+grad_len*grad_norm[0]+0.15, point[1]+grad_len*grad_norm[1]+0.15,
                r'$\nabla \log p$', fontsize=8, color=COLORS['red'])

# Tangent directions (perpendicular to gradient - these are good directions)
tangent = np.array([-grad_norm[1], grad_norm[0]])
tangent_len = 1.1

# Draw tangent arrows (teal, along contours)
ax3b_right.annotate('', xy=(point[0]+tangent_len*tangent[0], point[1]+tangent_len*tangent[1]),
                    xytext=(point[0], point[1]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['teal'], lw=2, mutation_scale=15))
ax3b_right.annotate('', xy=(point[0]-tangent_len*tangent[0], point[1]-tangent_len*tangent[1]),
                    xytext=(point[0], point[1]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['teal'], lw=2, mutation_scale=15))
ax3b_right.text(point[0]+tangent_len*tangent[0]+0.1, point[1]+tangent_len*tangent[1]-0.3,
                'along\ncontour', fontsize=7, color=COLORS['teal'], ha='left')

ax3b_right.set_xlabel(r'$\theta_1$')
ax3b_right.set_title('HMC: gradient reveals which directions explore', fontsize=8)
ax3b_right.set_xlim(-3, 3)
ax3b_right.set_ylim(-3, 3)
ax3b_right.set_aspect('equal')

fig3b.savefig('gradient_information.pdf')
print("Saved gradient_information.pdf")


# ============================================
# Figure 4: HMC vs Random Walk trajectories
# ============================================

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), constrained_layout=True)

# Correlated Gaussian target
cov = np.array([[1, 0.9], [0.9, 1]])
mean = np.array([0, 0])

def log_prob(theta):
    diff = theta - mean
    return -0.5 * diff @ np.linalg.inv(cov) @ diff

def grad_log_prob(theta):
    diff = theta - mean
    return -np.linalg.inv(cov) @ diff

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-0.5 * (np.linalg.inv(cov)[0,0]*X**2 + 2*np.linalg.inv(cov)[0,1]*X*Y + np.linalg.inv(cov)[1,1]*Y**2))

# Random walk MH
np.random.seed(123)
rw_chain = mh_sample(np.array([-2, 2]), 100, step_size=0.3)

# HMC simulation
def hmc_step(theta, step_size=0.1, n_leapfrog=20):
    p = np.random.randn(2)
    current_p = p.copy()
    current_theta = theta.copy()

    # Leapfrog integration
    p = p + 0.5 * step_size * grad_log_prob(theta)
    for _ in range(n_leapfrog - 1):
        theta = theta + step_size * p
        p = p + step_size * grad_log_prob(theta)
    theta = theta + step_size * p
    p = p + 0.5 * step_size * grad_log_prob(theta)

    # MH correction
    current_H = -log_prob(current_theta) + 0.5 * np.sum(current_p**2)
    proposed_H = -log_prob(theta) + 0.5 * np.sum(p**2)

    if np.random.rand() < np.exp(current_H - proposed_H):
        return theta
    return current_theta

np.random.seed(123)
hmc_chain = [np.array([-2, 2])]
for _ in range(100):
    hmc_chain.append(hmc_step(hmc_chain[-1], step_size=0.15, n_leapfrog=15))
hmc_chain = np.array(hmc_chain)

# Plot random walk
ax4a.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6)
ax4a.contour(X, Y, Z, levels=4, colors='white', linewidths=0.5, alpha=0.5)
ax4a.plot(rw_chain[:, 0], rw_chain[:, 1], '-', color=COLORS['orange'], alpha=0.5, linewidth=0.8)
ax4a.scatter(rw_chain[:, 0], rw_chain[:, 1], s=10, c=COLORS['orange'], zorder=5, edgecolors='white', linewidths=0.2)
ax4a.set_title('Random walk Metropolis', fontsize=9)
ax4a.set_xlabel(r'$\theta_1$')
ax4a.set_ylabel(r'$\theta_2$')
ax4a.set_xlim(-3, 3)
ax4a.set_ylim(-3, 3)

# Plot HMC
ax4b.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6)
ax4b.contour(X, Y, Z, levels=4, colors='white', linewidths=0.5, alpha=0.5)
ax4b.plot(hmc_chain[:, 0], hmc_chain[:, 1], '-', color=COLORS['teal'], alpha=0.5, linewidth=0.8)
ax4b.scatter(hmc_chain[:, 0], hmc_chain[:, 1], s=10, c=COLORS['teal'], zorder=5, edgecolors='white', linewidths=0.2)
ax4b.set_title('Hamiltonian Monte Carlo', fontsize=9)
ax4b.set_xlabel(r'$\theta_1$')
ax4b.set_xlim(-3, 3)
ax4b.set_ylim(-3, 3)

fig4.savefig('hmc_vs_rw.pdf')
print("Saved hmc_vs_rw.pdf")


# ============================================
# Figure 4b: VI geometry (Blei-style diagram)
# ============================================

from matplotlib.patches import Ellipse

fig4b, ax4b = plt.subplots(figsize=(SINGLE_COL * 1.5, 2.6), constrained_layout=True)

# Tilted ellipse for variational family
ellipse = Ellipse((0, 0), width=5.5, height=3.0, angle=-15,
                   fill=False, edgecolor='#333333', linewidth=1.8)
ax4b.add_patch(ellipse)

# Family label - inside, upper left
ax4b.text(-1.6, 0.8, r'$q(\theta; \phi)$', fontsize=12, color='#333333', style='italic')

# True posterior - outside ellipse, upper right
p_x, p_y = 3.8, 2.2
ax4b.plot(p_x, p_y, 'o', color='black', markersize=12, zorder=10)
ax4b.text(p_x - 1.4, p_y + 0.35, r'$p(\theta \mid x)$', fontsize=11)

# Optimal point - on the ellipse edge toward true posterior
q_x, q_y = 2.1, 1.05
ax4b.plot(q_x, q_y, 'o', color='black', markersize=12, zorder=10)
ax4b.text(q_x + 0.25, q_y - 0.1, r'$\phi^*$', fontsize=11)

# Initial point - inside ellipse, lower left
init_x, init_y = -1.3, -0.9
ax4b.plot(init_x, init_y, 'o', color='black', markersize=9, zorder=10)
ax4b.text(init_x - 0.25, init_y - 0.5, r'$\phi^{\mathrm{init}}$', fontsize=10, ha='center')

# Optimization path - simple straight line
ax4b.plot([init_x, q_x], [init_y, q_y], '-', color='#555555', linewidth=1.3)

# KL line from optimal to true posterior (dashed)
ax4b.plot([q_x, p_x], [q_y, p_y], '--', color='#555555', linewidth=1.5)

# KL label
ax4b.text(3.4, 1.35, r'KL$(q^* \| p)$', fontsize=10, color='#444444')

ax4b.set_xlim(-4, 5)
ax4b.set_ylim(-2.5, 3.2)
ax4b.set_aspect('equal')
ax4b.axis('off')

fig4b.savefig('vi_geometry.pdf', bbox_inches='tight', pad_inches=0.05)
print("Saved vi_geometry.pdf")


# ============================================
# Figure 5: Mode-seeking vs mass-covering
# ============================================

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), constrained_layout=True)

# Bimodal target
x = np.linspace(-4, 6, 500)
p = 0.4 * stats.norm.pdf(x, -1, 0.8) + 0.6 * stats.norm.pdf(x, 3, 1.2)

# KL(q||p) minimization: mode-seeking (q concentrates on one mode)
q_mode_seeking = stats.norm.pdf(x, 3, 0.8)

# KL(p||q) minimization: mass-covering (q spreads to cover both)
q_mass_covering = stats.norm.pdf(x, 1.5, 2.5)

ax5a.fill_between(x, p, alpha=0.3, color=COLORS['blue'], label=r'True $p(\theta \mid x)$')
ax5a.plot(x, p, color=COLORS['blue'], linewidth=1.5)
ax5a.plot(x, q_mode_seeking, color=COLORS['orange'], linewidth=2, linestyle='--', label=r'VI approximation $q(\theta)$')
ax5a.set_xlabel(r'$\theta$')
ax5a.set_ylabel('Density')
ax5a.set_title(r'VI minimizes KL$(q \| p)$: mode-seeking', fontsize=9)
ax5a.legend(fontsize=7, loc='upper right')
ax5a.set_xlim(-4, 6)

ax5b.fill_between(x, p, alpha=0.3, color=COLORS['blue'], label=r'True $p(\theta \mid x)$')
ax5b.plot(x, p, color=COLORS['blue'], linewidth=1.5)
ax5b.plot(x, q_mass_covering, color=COLORS['teal'], linewidth=2, linestyle='--', label=r'Alternative $q(\theta)$')
ax5b.set_xlabel(r'$\theta$')
ax5b.set_title(r'KL$(p \| q)$: mass-covering', fontsize=9)
ax5b.legend(fontsize=7, loc='upper right')
ax5b.set_xlim(-4, 6)

fig5.savefig('mode_seeking.pdf')
print("Saved mode_seeking.pdf")


# ============================================
# Figure 6: ELBO decomposition
# ============================================

fig6, ax6 = plt.subplots(figsize=(SINGLE_COL * 1.3, 2.5), constrained_layout=True)

# Schematic bar chart showing ELBO decomposition
categories = [r'log $p(x)$', 'ELBO', r'KL$(q \| p)$']
log_evidence = 10
elbo = 7
kl = 3

bars = ax6.barh([0], [log_evidence], color=COLORS['blue'], alpha=0.7, label='Log evidence (constant)')
bars = ax6.barh([1], [elbo], color=COLORS['teal'], alpha=0.7, label='ELBO (maximize this)')
bars = ax6.barh([1], [kl], left=elbo, color=COLORS['orange'], alpha=0.7, label='KL gap (want this small)')

ax6.set_yticks([0, 1])
ax6.set_yticklabels([r'$\log p(x)$', 'Decomposition'])
ax6.set_xlabel('Value')
ax6.legend(fontsize=7, loc='lower right')
ax6.set_title(r'$\log p(x) = \mathrm{ELBO} + \mathrm{KL}(q \| p)$', fontsize=9)

# Add annotation
ax6.annotate('', xy=(elbo, 1.3), xytext=(elbo + kl, 1.3),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax6.text(elbo + kl/2, 1.45, 'Gap', ha='center', fontsize=8)

fig6.savefig('elbo_decomposition.pdf')
print("Saved elbo_decomposition.pdf")


# ============================================
# Figure 7: Trace plots (good vs bad)
# ============================================

fig7, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 3), constrained_layout=True)

n_iter = 500

# Good trace: well-mixed
np.random.seed(42)
good_trace = np.cumsum(np.random.randn(n_iter) * 0.3)
good_trace = good_trace - np.mean(good_trace) + 2  # Center around 2

# Bad trace: not converged, trending
bad_trace = np.linspace(-2, 2, n_iter) + np.cumsum(np.random.randn(n_iter) * 0.1)

# Bad trace: stuck
np.random.seed(99)
stuck_trace = np.zeros(n_iter)
stuck_trace[:200] = -1 + np.random.randn(200) * 0.1
stuck_trace[200:350] = 1 + np.random.randn(150) * 0.1
stuck_trace[350:] = -0.5 + np.random.randn(150) * 0.1

axes[0, 0].plot(good_trace, color=COLORS['teal'], linewidth=0.5)
axes[0, 0].axhline(np.mean(good_trace), color=COLORS['orange'], linestyle='--', linewidth=1)
axes[0, 0].set_title('Good: stationary noise', fontsize=9)
axes[0, 0].set_ylabel(r'$\theta$')

axes[0, 1].plot(bad_trace, color=COLORS['red'], linewidth=0.5)
axes[0, 1].set_title('Bad: trending (not converged)', fontsize=9)

axes[1, 0].plot(stuck_trace, color=COLORS['red'], linewidth=0.5)
axes[1, 0].set_title('Bad: stuck in modes', fontsize=9)
axes[1, 0].set_ylabel(r'$\theta$')
axes[1, 0].set_xlabel('Iteration')

# Autocorrelation comparison
from numpy.fft import fft, ifft
def autocorr(x):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[n-1:] / result[n-1]

acf_good = autocorr(good_trace)[:100]
acf_bad = autocorr(stuck_trace)[:100]

axes[1, 1].plot(acf_good, color=COLORS['teal'], linewidth=1.5, label='Good chain')
axes[1, 1].plot(acf_bad, color=COLORS['red'], linewidth=1.5, label='Stuck chain')
axes[1, 1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
axes[1, 1].set_title('Autocorrelation', fontsize=9)
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('Autocorrelation')
axes[1, 1].legend(fontsize=7)
axes[1, 1].set_xlim(0, 100)

fig7.savefig('trace_plots.pdf')
print("Saved trace_plots.pdf")


# ============================================
# Figure 8: VI optimization - real ELBO trace
# ============================================

# Load real ELBO trace from NumPyro SVI on Bayesian linear regression
# (generated separately: 5-feature regression, 200 data points, mean-field guide)
elbo_trace = np.load('vi_elbo_trace.npy')

fig8, ax8 = plt.subplots(figsize=(SINGLE_COL, 2.5), constrained_layout=True)

iterations = np.arange(len(elbo_trace))
ax8.plot(iterations, elbo_trace, color=COLORS['teal'], linewidth=1.0, alpha=0.8)

# Add smoothed version
from scipy.ndimage import uniform_filter1d
elbo_smooth = uniform_filter1d(elbo_trace, size=20)
ax8.plot(iterations, elbo_smooth, color=COLORS['blue'], linewidth=2, label='ELBO (smoothed)')

ax8.set_xlabel('Optimization step')
ax8.set_ylabel('ELBO')
ax8.set_title('VI optimization on Bayesian linear regression', fontsize=9)
ax8.legend(fontsize=7, loc='lower right')
ax8.set_xlim(0, len(elbo_trace))

fig8.savefig('vi_optimization.pdf')
print("Saved vi_optimization.pdf")


# ============================================
# Figure 9: Corner plot - Planck cosmological parameters
# ============================================

from getdist import loadMCSamples
import corner

# Load actual Planck PR4 MCMC chains
# These are from Lemos & Lewis (2023), arXiv:2302.12911
# Data: Planck PR4 TT,TE,EE + lowE + lensing
planck = loadMCSamples('PlanckEarlyLCDM/spline_planck_PR4_TTTEEE_lowE_lensing_ISW')

# Select key cosmological parameters for visualization
param_names = ['H0', 'omegam', 'sigma8', 'ns']
param_labels = [
    r'$H_0$ [km/s/Mpc]',
    r'$\Omega_m$',
    r'$\sigma_8$',
    r'$n_s$'
]

# Extract samples with weights
samples_array = np.column_stack([planck.getParams().__dict__[p] for p in param_names])
weights = planck.weights

# Resample to get unweighted samples for corner
n_resample = 8000
indices = np.random.choice(len(weights), size=n_resample, p=weights/weights.sum())
samples_resampled = samples_array[indices]

# Create corner plot
fig9 = corner.corner(
    samples_resampled,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f',
    title_kwargs={"fontsize": 8},
    label_kwargs={"fontsize": 9},
    color=COLORS['blue'],
    plot_datapoints=True,
    plot_density=True,
    fill_contours=True,
    levels=[0.68, 0.95],
    hist_kwargs={'linewidth': 1.5, 'density': True, 'color': COLORS['blue']},
    contour_kwargs={'linewidths': 1.0},
    data_kwargs={'alpha': 0.5, 'ms': 1.2, 'color': COLORS['blue']},
)

fig9.set_size_inches(DOUBLE_COL, DOUBLE_COL)
fig9.savefig('corner_cosmology.pdf', bbox_inches='tight')
print("Saved corner_cosmology.pdf")


print("\nAll Chapter 2 figures generated successfully!")
