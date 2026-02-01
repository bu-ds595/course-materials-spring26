"""
Banana-shaped 2D posterior contour plot for MCMC visualization.
Shows posterior mean (red star) and credible region (orange contour).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

# Colors
RED = '#CC3311'
GREEN = '#009988'

# Column widths
DOUBLE_COL = 6.75

# Banana distribution: p(theta) ~ exp(-(theta_1^2 + (theta_2 - theta_1^2)^2))
def banana_density(theta1, theta2):
    return np.exp(-(theta1**2 + (theta2 - theta1**2)**2))

# Create grid
theta1 = np.linspace(-3, 3, 300)
theta2 = np.linspace(-1.5, 5, 300)
T1, T2 = np.meshgrid(theta1, theta2)
Z = banana_density(T1, T2)

# Compute posterior mean via numerical integration
dtheta = (theta1[1] - theta1[0]) * (theta2[1] - theta2[0])
Z_norm = Z / (Z.sum() * dtheta)
mean_theta1 = (T1 * Z_norm).sum() * dtheta
mean_theta2 = (T2 * Z_norm).sum() * dtheta

# Create figure
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5), constrained_layout=True)

# Plot filled contours
levels = np.linspace(0.01, Z.max(), 10)
cf = ax.contourf(T1, T2, Z, levels=levels, cmap='Blues')

# Add contour lines
ax.contour(T1, T2, Z, levels=levels[::2], colors='steelblue', linewidths=0.6, alpha=0.6)

# Add credible region contour (orange) - roughly 95% credible region
credible_level = Z.max() * 0.1  # Approximate level for credible region
ax.contour(T1, T2, Z, levels=[credible_level], colors=[GREEN], linewidths=2.5)

# Add posterior mean as red star
ax.plot(mean_theta1, mean_theta2, '*', color=RED, markersize=20, markeredgecolor='white', markeredgewidth=0.8)

# Labels
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1, 4)

# Save
fig.savefig(Path(__file__).parent / 'banana_posterior.png', dpi=200)
fig.savefig(Path(__file__).parent / 'banana_posterior.pdf')
plt.close()
print(f"Posterior mean: ({mean_theta1:.2f}, {mean_theta2:.2f})")
print("Saved banana_posterior.png and banana_posterior.pdf")
