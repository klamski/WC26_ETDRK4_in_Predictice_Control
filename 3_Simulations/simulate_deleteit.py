#%% #################### SIMULATE GROUND TRUTH ##################################
# Simulate via stiff ode solver
# 1. Define the basic pattern (the single column)
column_pattern = np.array([0, 7.5, 0, 0.1])
column_vector = column_pattern.reshape(-1, 1) # Shape (4, 1)
np_array_alt = np.tile(column_vector, (1, 48))
u_opt = np_array_alt

t_span = (0, N * Delta)
t_eval = np.arange(0, t_span[1] + 1, Delta)

t_data = Delta * np.arange(t0, t0 + N)
u_func = scipy.interpolate.interp1d(
    t_data, u_opt, kind="previous", bounds_error=False, fill_value="extrapolate"
)
d_func = scipy.interpolate.interp1d(
    t_data,
    d_cl[:, t0 : (t0 + N)],
    kind="previous",
    bounds_error=False,
    fill_value="extrapolate",
)


def ode_wrapper(t, x):
    u = u_func(t)
    d = d_func(t)
    M = ca_model.odes_screen_lin(params)
    A = M*u[3]
    nonlin = model.odes_screen_nonlin(x, u, d, params)
    return A @ x + nonlin


sol = scipy.integrate.solve_ivp(
    ode_wrapper,
    t_span,
    x0,
    method="BDF",
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8,
)

y_sim = h(sol["y"]).full()


import matplotlib.pyplot as plt
import numpy as np

# === CONTROL VARIABLES ===
font_size = 11
axis_width = 1.2
figsize = (12.5, 6)

# === GLOBAL STYLE SETTINGS ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "legend.fontsize": font_size - 1,
    "xtick.labelsize": font_size - 1,
    "ytick.labelsize": font_size - 1,
    "axes.linewidth": axis_width,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# === CREATE SUBPLOTS ===
fig, axs = plt.subplots(2, 5, figsize=figsize, sharex=True, dpi=300)
time = np.arange(t0, t0 + N + 1)

def beautify(ax):
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

# === TOP ROW ===
axs[0, 4].axis('on')
axs[0, 4].set_xticks([])
axs[0, 4].set_yticks([])
axs[0, 4].spines[:].set_visible(False)

# axs[0, 0].plot(time, y_opt[0, :], color="C0", label="Optimal")
axs[0, 0].plot(time, y_sim[0, :], color="C0", linestyle="solid")
axs[0, 0].set_title("Dry Weight [g/m^2]")
beautify(axs[0, 0])

axs[0, 1].axhline(1600, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
# axs[0, 1].plot(time, y_opt[1, :] * 1e3, color="C0", label="Below")
axs[0, 1].plot(time, y_sim[1, :] * 1e3, color="C0", linestyle="solid")
# axs[0, 1].plot(time, y_opt[4, :] * 1e3, color="C1", label="Above")
axs[0, 1].plot(time, y_sim[4, :] * 1e3, color="C1", linestyle="solid")
axs[0, 1].set_title("CO$_2$ Concentration [ppm]")
# axs[0, 1].set_ylim(300, 2000)
# axs[0, 1].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 1])

axs[0, 2].axhline(30, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 2].axhline(10, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
# axs[0, 2].plot(time, y_opt[2, :], color="C0", label="Below")
axs[0, 2].plot(time, y_sim[2, :], color="C0", linestyle="solid")
# axs[0, 2].plot(time, y_opt[5, :], color="C1", label="Above")
axs[0, 2].plot(time, y_sim[5, :], color="C1", linestyle="solid")
axs[0, 2].plot(time, d_cl[2, : N + 1], color="C4", label="Out")
axs[0, 2].set_title("Temperature [^oC]")
# axs[0, 2].set_ylim(0, 40)
# axs[0, 2].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 2])

rh_out = (
    params[11]
    * (d_cl[2, : N + 1] + params[12])
    / (11 * np.exp(params[26] * d_cl[2, : N + 1] / (d_cl[2, : N + 1] + params[27])))
    * d_cl[3, : N + 1]
    * 1e2
)
axs[0, 3].axhline(80, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
# axs[0, 3].plot(time, y_opt[3, :], color="C0", label="Below")
axs[0, 3].plot(time, y_sim[3, :], color="C0", linestyle="solid")
# axs[0, 3].plot(time, y_opt[6, :], color="C1", label="Above")
axs[0, 3].plot(time, y_sim[6, :], color="C1", linestyle="solid")
axs[0, 3].plot(time, rh_out, color="C4", label="Out")
# axs[0, 3].set_ylim(50, 200)
axs[0, 3].set_title("Relative Humidity [%]")
# axs[0, 3].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 3])

# Add a global legend specifically in axes[1, 3]
handles = [
    plt.Line2D([], [], color='C0', linewidth=2, label='Below'),
    plt.Line2D([], [], color='C1', linewidth=2, label="Above"), 
    plt.Line2D([], [], color='C4', linewidth=2, label="Out"),
    # plt.Line2D([], [], color='k', linestyle="solid" , label="Ground Truth"),
    plt.Line2D([], [], color="k", linestyle="--", label="Limits")
]

# Turn off the axes[1, 3] subplot and use it for the legend
axs[0, 4].axis('off')  # Disable the plot area of axes[1, 3]
axs[0, 4].legend(handles=handles, loc='center', fontsize=12, frameon=False)  # Add legend to the center

# === BOTTOM ROW ===
axs[1, 0].plot(time, d_cl[0, : N + 1] / 1e3, color="C4")
axs[1, 0].set_title("Irradiance [W/m^2]")
axs[1, 0].set_xlabel("Time (t)")
beautify(axs[1, 0])

axs[1, 1].plot(time[:-1], u_opt[0, :], drawstyle="steps-post", color="C2")
axs[1, 1].set_title("CO$_2$ Supply [mg/m^2/s]")
axs[1, 1].set_xlabel("Time (t)")
beautify(axs[1, 1])

axs[1, 2].plot(time[:-1], u_opt[2, :], drawstyle="steps-post", color="C2")
axs[1, 2].set_title("Heating [W/m^2]")
axs[1, 2].set_xlabel("Time (t)")
beautify(axs[1, 2])

axs[1, 3].plot(time[:-1], u_opt[1, :], drawstyle="steps-post", color="C2")
axs[1, 3].set_title("Ventilation [L/m^2/s]")
axs[1, 3].set_xlabel("Time (t)")
beautify(axs[1, 3])

axs[1, 4].plot(time[:-1], u_opt[3, :], drawstyle="steps-post", color="C2")
axs[1, 4].set_title(r"Flow ($\alpha$) [flow]")
axs[1, 4].set_xlabel("Time (t)")
beautify(axs[1, 4])

# === CENTRALIZED LABEL BOX ===
# axs[0, 4].text(0.05, 0.85, "Y-Axis Labels:", fontsize=font_size + 1, fontweight="bold")

# labels = [
#     ("Dry weight", "g m$^{-2}$"),
#     ("CO$_2$", "ppm"),
#     ("Temperature", r"$^\circ$C"),
#     ("Rel. Humidity", "\%"),
#     ("Irradiance", "kW m$^{-2}$"),
#     ("CO$_2$ Supply", "mg m$^{-2}$ s$^{-1}$"),
#     ("Heating", "W m$^{-2}$"),
#     ("Ventilation", "L m$^{-2}$ s$^{-1}$"),
#     ("Flow", r"(unitless or custom)")
# ]

# for i, (name, unit) in enumerate(labels):
#     axs[0, 4].text(
#         0.05,
#         0.75 - i * 0.085,
#         f"{name}: {unit}",
#         fontsize=font_size
#     )
fig.suptitle(f"Integration method: {integration_method}", fontsize=font_size + 3, fontweight='bold')

# === FINAL LAYOUT ===
plt.tight_layout(pad=1.5)
plt.subplots_adjust(wspace=0.5)
plt.show()
