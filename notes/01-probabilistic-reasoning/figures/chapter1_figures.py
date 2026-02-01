"""
Publication-quality figures for Chapter 1: Probabilistic Reasoning
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.special import factorial

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

np.random.seed(21)

# ============================================
# Figure 1: Profile Likelihood
# ============================================
# For line fitting, show profile likelihood over slope m
# (profiling out the intercept c)

# Generate data
N = 20
x = np.linspace(0, 5, N)
m_true, c_true, sigma = 2.5, 1.0, 1.5
y = m_true * x + c_true + np.random.normal(0, sigma, N)

def neg_log_likelihood(m, c, x, y, sigma):
    """Negative log-likelihood for line fitting."""
    residuals = y - (m * x + c)
    return 0.5 * np.sum(residuals**2) / sigma**2

def profile_likelihood(m_vals, x, y, sigma):
    """For each m, find optimal c and return the log-likelihood."""
    profile = []
    c_opts = []
    for m in m_vals:
        # Optimal c for given m: c = mean(y - m*x)
        c_opt = np.mean(y - m * x)
        c_opts.append(c_opt)
        nll = neg_log_likelihood(m, c_opt, x, y, sigma)
        profile.append(-nll)  # Return log-likelihood
    return np.array(profile), np.array(c_opts)

m_range = np.linspace(1.5, 3.5, 200)
profile_ll, c_profile = profile_likelihood(m_range, x, y, sigma)
profile_ll_normalized = profile_ll - profile_ll.max()  # Normalize to 0 at max

# MLE
m_mle_idx = np.argmax(profile_ll)
m_mle = m_range[m_mle_idx]

fig1, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.2), constrained_layout=True)

ax1.plot(m_range, profile_ll_normalized, '-', color=COLORS['blue'], linewidth=1.5)
ax1.axvline(m_mle, color=COLORS['orange'], linestyle='--', linewidth=1, label='Max. likelihood')
ax1.axvline(m_true, color=COLORS['gray'], linestyle=':', linewidth=1, label='True')

ax1.set_xlabel(r'Slope $m$')
ax1.set_ylabel(r'$\ell(m) - \ell(\hat{m})$')
ax1.set_ylim(-4, 0.5)
ax1.legend(loc='lower left', frameon=True)

fig1.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/profile_likelihood.pdf')
print("Saved profile_likelihood.pdf")


# ============================================
# Figure 2: Common Distributions
# ============================================

fig2, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.5), constrained_layout=True)

# Gaussian
ax = axes[0, 0]
x_gauss = np.linspace(-4, 4, 200)
for mu, sigma, label in [(0, 1, r'$\mu=0, \sigma=1$'),
                          (0, 0.5, r'$\mu=0, \sigma=0.5$'),
                          (1, 1.5, r'$\mu=1, \sigma=1.5$')]:
    ax.plot(x_gauss, stats.norm.pdf(x_gauss, mu, sigma), linewidth=1.2, label=label)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$p(x)$')
ax.set_title('Gaussian')
ax.legend(fontsize=6, frameon=False)
ax.set_xlim(-4, 4)

# Poisson
ax = axes[0, 1]
n_vals = np.arange(0, 20)
for lam, color in [(2, COLORS['blue']), (5, COLORS['orange']), (10, COLORS['teal'])]:
    pmf = stats.poisson.pmf(n_vals, lam)
    ax.bar(n_vals + 0.1*(lam-5), pmf, width=0.25, alpha=0.8, color=color,
           label=rf'$\lambda={lam}$')
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$p(n)$')
ax.set_title('Poisson')
ax.legend(fontsize=6, frameon=False)
ax.set_xlim(-0.5, 18)

# Binomial
ax = axes[0, 2]
k_vals = np.arange(0, 21)
n = 20
for p, color in [(0.2, COLORS['blue']), (0.5, COLORS['orange']), (0.8, COLORS['teal'])]:
    pmf = stats.binom.pmf(k_vals, n, p)
    ax.bar(k_vals + 0.3*(p-0.5), pmf, width=0.25, alpha=0.8, color=color,
           label=rf'$p={p}$')
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$p(k)$')
ax.set_title(f'Binomial ($n={n}$)')
ax.legend(fontsize=6, frameon=False)

# Categorical (bar chart)
ax = axes[1, 0]
categories = ['A', 'B', 'C', 'D']
probs = [0.4, 0.3, 0.2, 0.1]
bars = ax.bar(categories, probs, color=[COLORS['blue'], COLORS['orange'],
                                         COLORS['teal'], COLORS['red']], alpha=0.8)
ax.set_xlabel('Category')
ax.set_ylabel(r'$\pi_k$')
ax.set_title('Categorical')
ax.set_ylim(0, 0.5)

# Uniform
ax = axes[1, 1]
x_unif = np.linspace(-1, 4, 500)
for a, b, color in [(0, 1, COLORS['blue']), (0, 2, COLORS['orange']), (1, 3, COLORS['teal'])]:
    pdf = np.where((x_unif >= a) & (x_unif <= b), 1/(b-a), 0)
    ax.plot(x_unif, pdf, linewidth=1.5, color=color, label=rf'$[{a},{b}]$')
    ax.fill_between(x_unif, pdf, alpha=0.2, color=color)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$p(x)$')
ax.set_title('Uniform')
ax.legend(fontsize=6, frameon=False)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 1.3)

# Remove the empty subplot in position [1, 2]
axes[1, 2].axis('off')

fig2.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/common_distributions.pdf')
print("Saved common_distributions.pdf")


# ============================================
# Figure 3: Occam's Razor / Model Complexity
# ============================================

fig3, ax3 = plt.subplots(figsize=(SINGLE_COL, 2.5), constrained_layout=True)

# Create synthetic "evidence" curves for simple vs complex models
x_data = np.linspace(0, 10, 500)

# Simple model: concentrated around one region
simple_evidence = stats.norm.pdf(x_data, 4, 1.0)
simple_evidence = simple_evidence / simple_evidence.max() * 0.8

# Complex model: spread over broader range
complex_evidence = stats.norm.pdf(x_data, 5, 3.0)
complex_evidence = complex_evidence / complex_evidence.max() * 0.35

ax3.fill_between(x_data, simple_evidence, alpha=0.4, color=COLORS['blue'], label='Simple model')
ax3.plot(x_data, simple_evidence, color=COLORS['blue'], linewidth=1.5)

ax3.fill_between(x_data, complex_evidence, alpha=0.4, color=COLORS['magenta'], label='Complex model')
ax3.plot(x_data, complex_evidence, color=COLORS['magenta'], linewidth=1.5)

# Mark observed data point
x_obs = 4.5
ax3.axvline(x_obs, color='black', linestyle='--', linewidth=1, label='Observed data')
ax3.plot(x_obs, simple_evidence[np.argmin(np.abs(x_data - x_obs))], 'o',
         color=COLORS['blue'], markersize=6, zorder=5)
ax3.plot(x_obs, complex_evidence[np.argmin(np.abs(x_data - x_obs))], 'o',
         color=COLORS['magenta'], markersize=6, zorder=5)

ax3.set_xlabel('Possible datasets')
ax3.set_ylabel(r'$p(\mathrm{data} \mid \mathcal{M})$')
ax3.legend(loc='upper right', frameon=True, fontsize=7)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 1.0)
ax3.set_xticks([])  # Remove x ticks since it's conceptual

fig3.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/model_complexity.pdf')
print("Saved model_complexity.pdf (updated)")


# ============================================
# Figure 4: Bayes' Theorem Visualization
# ============================================

fig4, axes4 = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.0), constrained_layout=True)

theta = np.linspace(-2, 6, 500)

# Prior: broad Gaussian
prior = stats.norm.pdf(theta, 2, 1.5)
prior = prior / prior.max()

# Likelihood: peaks at data-informed value
likelihood = stats.norm.pdf(theta, 3.5, 0.8)
likelihood = likelihood / likelihood.max()

# Posterior: combination (narrower, shifted)
posterior = prior * likelihood
posterior = posterior / posterior.max()

# Panel (a): Prior
ax = axes4[0]
ax.fill_between(theta, prior, alpha=0.4, color=COLORS['blue'])
ax.plot(theta, prior, color=COLORS['blue'], linewidth=1.5)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Density')
ax.set_title(r'(a) Prior $p(\theta)$')
ax.set_xlim(-1, 6)
ax.set_ylim(0, 1.15)

# Panel (b): Likelihood
ax = axes4[1]
ax.fill_between(theta, likelihood, alpha=0.4, color=COLORS['orange'])
ax.plot(theta, likelihood, color=COLORS['orange'], linewidth=1.5)
ax.set_xlabel(r'$\theta$')
ax.set_title(r'(b) Likelihood $p(x \mid \theta)$')
ax.set_xlim(-1, 6)
ax.set_ylim(0, 1.15)

# Panel (c): Posterior
ax = axes4[2]
ax.fill_between(theta, posterior, alpha=0.4, color=COLORS['teal'])
ax.plot(theta, posterior, color=COLORS['teal'], linewidth=1.5)
# Show prior and likelihood faintly for reference
ax.plot(theta, prior, '--', color=COLORS['blue'], linewidth=0.8, alpha=0.5)
ax.plot(theta, likelihood, '--', color=COLORS['orange'], linewidth=0.8, alpha=0.5)
ax.set_xlabel(r'$\theta$')
ax.set_title(r'(c) Posterior $p(\theta \mid x)$')
ax.set_xlim(-1, 6)
ax.set_ylim(0, 1.15)

fig4.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/bayes_theorem.pdf')
print("Saved bayes_theorem.pdf (updated)")


# ============================================
# Regenerate line fitting figures with consistent style
# ============================================

# True parameters
m_true = 2.5
c_true = 1.0
sigma = 1.5

# Generate data (t = input, x = response/data)
np.random.seed(21)
N = 20
t = np.linspace(0, 5, N)
x_true = m_true * t + c_true
x_data = x_true + np.random.normal(0, sigma, N)

# MLE fit (OLS)
T = np.column_stack([t, np.ones(N)])
theta_mle = np.linalg.lstsq(T, x_data, rcond=None)[0]
m_mle, c_mle = theta_mle

# Posterior covariance
TtT = T.T @ T
cov_post = sigma**2 * np.linalg.inv(TtT)
mean_post = theta_mle

# Figure 5: Data with fitted line (with error bars)
fig5, ax5 = plt.subplots(figsize=(SINGLE_COL, 2.5), constrained_layout=True)

# Plot data with error bars showing sigma
ax5.errorbar(t, x_data, yerr=sigma, fmt='o', color=COLORS['blue'], markersize=4,
             capsize=2, capthick=0.8, elinewidth=0.8, zorder=3, label='Data')
t_line = np.linspace(-0.2, 5.2, 100)
ax5.plot(t_line, m_true * t_line + c_true, '--', color=COLORS['gray'],
         linewidth=1.2, label='True', zorder=2)
ax5.plot(t_line, m_mle * t_line + c_mle, '-', color=COLORS['orange'],
         linewidth=1.5, label='Max. likelihood', zorder=2)

ax5.set_xlabel(r'$t$')
ax5.set_ylabel(r'$x$')
ax5.set_xlim(-0.2, 5.2)
ax5.legend(loc='upper left', frameon=True)

fig5.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/line_fit_data.pdf')
print("Saved line_fit_data.pdf (updated)")


# Figure 6: Combined (data + posterior)
m_range = np.linspace(1.5, 3.5, 200)
c_range = np.linspace(-1.0, 3.0, 200)
M, C = np.meshgrid(m_range, c_range)
pos = np.dstack((M, C))
rv = stats.multivariate_normal(mean_post, cov_post)
Z = rv.pdf(pos)

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5), constrained_layout=True)

# Left panel: Data with posterior samples (with error bars)
ax6a.errorbar(t, x_data, yerr=sigma, fmt='o', color=COLORS['blue'], markersize=4,
              capsize=2, capthick=0.8, elinewidth=0.8, zorder=3, label='Data')

np.random.seed(123)
for i in range(30):
    m_samp, c_samp = np.random.multivariate_normal(mean_post, cov_post)
    ax6a.plot(t_line, m_samp * t_line + c_samp, color=COLORS['teal'],
              alpha=0.15, linewidth=0.8, zorder=1)

# Dummy line for legend
ax6a.plot([], [], color=COLORS['teal'], alpha=0.5, linewidth=1.5,
          label=r'Samples from $p(\theta \mid x)$')

ax6a.plot(t_line, m_true * t_line + c_true, '--', color='black',
         linewidth=1.2, label='True', zorder=2)

ax6a.set_xlabel(r'$t$')
ax6a.set_ylabel(r'$x$')
ax6a.set_xlim(-0.2, 5.2)
ax6a.set_title('(a) Data + posterior samples')
ax6a.legend(loc='upper left', frameon=True)

# Right panel: 2D posterior
cf = ax6b.contourf(M, C, Z / Z.max(), levels=np.linspace(0, 1, 15),
                    cmap='Blues', alpha=0.8)
ax6b.contour(M, C, Z / Z.max(), levels=[0.135, 0.607],
             colors=[COLORS['blue'], COLORS['blue']],
             linewidths=[1.0, 1.0], linestyles=['--', '-'])
ax6b.scatter([m_true], [c_true], marker='*', s=80, color=COLORS['orange'],
             zorder=5, label='True')

ax6b.set_xlabel(r'Slope $m$')
ax6b.set_ylabel(r'Intercept $c$')
ax6b.set_title('(b) Joint posterior')
ax6b.legend(loc='upper right', frameon=True)

fig6.savefig('/Users/smsharma/Projects/ds595-ai4science/notes/chapters/01-probabilistic-reasoning/figures/line_regression_combined.pdf')
print("Saved line_regression_combined.pdf (updated)")

print("\nAll figures generated successfully!")
