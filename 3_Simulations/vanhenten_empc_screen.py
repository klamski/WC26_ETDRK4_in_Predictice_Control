import numpy as np
import scipy.io
import scipy.interpolate
import scipy.integrate
import casadi
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory of ../2_Models to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_Models")))
import etd
import vanhenten_model as model

# Open the file in read mode
data = scipy.io.loadmat("../1_InputData/weather.mat")
weather = {k: data[k] for k in ["co2", "hum", "iGlob", "tOut"]}
params = model.params

Delta = 30 * 60  # 30 minutes
N = 2 * 24 * 2  # 2 days with 30 min time steps
alpha = 0.001#0.001  # Air exchange flux between top and bottom of screen (m/s)

# Change choose air exchange flux anywhere between 100 to 0.001, but starts to get shaky around 0.001.
# But for 0.001, we no longer have a stiff system, and instead we are running into issues with controller.
# Basically, we can no longer control the humidity properly beause of how slow the transfer is from below to above the screen.


nx = 7
nu = 3
nd = 4

t0 = 0
x0 = np.array([0.0035, 0.001, 15, 0.008, 0.002, 20, 0.012])

c_co2 = 1e-6 * 0.42 * Delta
# per kg/s of CO2
c_q = 6.35e-9 * Delta
# per W of heat
c_dw = -16
# price per kg of dry weight


d_cl = np.array([weather["iGlob"], weather["co2"], weather["tOut"], weather["hum"]])

d_cl = d_cl[..., 0]


def interpolate_weather(d_cl, Delta):
    original_length = len(d_cl)
    original_indices = np.arange(original_length)

    step = Delta / 300  # E.g., 3.0 for 15-min from 5-min
    original_indices = np.arange(d_cl.shape[1])
    new_indices = np.arange(0, d_cl.shape[1], step)

    # Create an interpolator for each feature (row)
    interp_func = scipy.interpolate.interp1d(
        original_indices, d_cl, axis=1, kind="linear", fill_value="extrapolate"
    )

    # Interpolate
    return interp_func(new_indices)


d_cl = interpolate_weather(d_cl, Delta)


A = model.odes_screen_lin(params, alpha)


def f_etd(x, u, d):
    return etd.etdrk4(A, lambda x: model.odes_screen_nonlin(x, u, d, params), x, Delta)


x = casadi.SX.sym("x", nx)
u = casadi.SX.sym("u", nu)
d = casadi.SX.sym("d", nd)

f = casadi.Function("f", [x, u, d], [f_etd(x, u, d)], ["x", "u", "d"], ["xplus"])

h = casadi.Function("h", [x], [model.h_meas_screen(x, params)], ["x"], ["y"])

w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

x_out = []
u_out = []
y_out = []

d_p = casadi.MX.sym("d_p", nd, N)

lbx = [0, 1e-4, 5, 0, 0, 0, 0]
ubx = [600, 0.009, 50, 1, 0.009, 50, 1]
lbu = [0, 0, 0]
ubu = [1.2, 7.5, 150]

Xk = casadi.MX.sym("X_0", nx)
w.append(Xk)
lbw.append(lbx)
ubw.append(ubx)
w0.append(lbx)
x_out.append(Xk)
y_out.append(h(Xk))

for k in range(N):
    Uk = casadi.MX.sym("U_" + str(k), nu)
    w.append(Uk)
    lbw.append(lbu)
    ubw.append(ubu)
    w0.append(lbu)
    u_out.append(Uk)

    Xk_plus = f(x=Xk, u=Uk, d=d_p[:, k])["xplus"]

    J = J + c_dw * (Xk_plus[0] - Xk[0]) + c_q * Uk[2] + c_co2 * Uk[0]

    Xk = casadi.MX.sym("X_" + str(k + 1), nx)
    w.append(Xk)
    lbw.append(lbx)
    ubw.append(ubx)
    w0.append(lbx)
    x_out.append(Xk)
    y_out.append(h(Xk))

    g.append(Xk_plus - Xk)
    lbg.append(nx * [0])
    ubg.append(nx * [0])

    g.append(h(Xk))
    lbg.append([0, 0, 10, 0, 0, 0, 0])
    ubg.append([600, 1.6, 25, 80, 2.5, 40, 100])

opts = {
    "ipopt": {
        # "print_level": 0,  # IPOPT verbosity (0-12, where 5 is moderate)
        # "max_iter": 3000,  # Maximum number of iterations
        # "warm_start_init_point": "yes",
        # 'tol': 1e-6,           # Convergence tolerance
    },
    # "print_time": False,  # Print timing information
    # "verbose": False,  # CasADi-level verbosity
}

w = casadi.vertcat(*w)
g = casadi.vertcat(*g)
x_out = casadi.horzcat(*x_out)
u_out = casadi.horzcat(*u_out)
y_out = casadi.horzcat(*y_out)

prob = {
    "f": J,
    "x": w,
    "g": g,
    "p": d_p,
}
solver = casadi.nlpsol("solver", "ipopt", prob, opts)

trajectories = casadi.Function(
    "trajectories", [w], [x_out, u_out, y_out], ["w"], ["x", "u", "y"]
)

w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)


lbw[:nx] = x0
ubw[:nx] = x0
w0[:nx] = x0

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=d_cl[:, t0:N])

x_opt, u_opt, y_opt = trajectories(sol["x"])
x_opt = x_opt.full()
u_opt = u_opt.full()
y_opt = y_opt.full()


# Simulate via stiff ode solver

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
    return model.odes_screen(x, u, d, params, alpha)


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


# fig, axs = plt.subplots(2, 4, figsize=(9, 4.5), sharex=True, dpi=300)

# time = np.arange(t0, t0 + N + 1)

# ax = axs[0, 0]
# ax.plot(time, y_opt[0, :], color="C0")
# ax.plot(time, y_sim[0, :], color="k", linestyle="dotted")
# ax.set_xlim(0, N)
# ax.set_title("Dry weight (g/m$^2$)")

# ax = axs[0, 1]
# ax.axhline(1600, color="k", linestyle="dashed", alpha=0.5, linewidth=0.5)
# ax.plot(time, y_opt[1, :] * 1e3, color="C0", label="Below")
# ax.plot(time, y_sim[1, :] * 1e3, color="k", linestyle="dotted")
# ax.plot(time, y_opt[4, :] * 1e3, color="C1", label="Above")
# ax.plot(time, y_sim[4, :] * 1e3, color="k", linestyle="dotted")
# ax.set_title("CO$_2$ (ppm)")
# ax.set_ylim(300, 2000)
# ax.legend(loc=9, ncol=2)

# ax = axs[0, 2]
# ax.axhline(25, color="k", linestyle="dashed", alpha=0.5, linewidth=0.5)
# ax.axhline(10, color="k", linestyle="dashed", alpha=0.5, linewidth=0.5)
# ax.plot(time, y_opt[2, :], color="C0", label="Below")
# ax.plot(time, y_sim[2, :], color="k", linestyle="dotted")
# ax.plot(time, y_opt[5, :], color="C1", label="Above")
# ax.plot(time, y_sim[5, :], color="k", linestyle="dotted")
# ax.plot(time, d_cl[2, : N + 1], color="C4", label="Out")
# ax.set_title("Temp ($^o$C)")
# ax.set_ylim(0, 32)
# ax.legend(loc=9, ncol=2)

# ax = axs[0, 3]
# # ax.axhline(100,color="k")
# ax.axhline(80, color="k", linestyle="dashed", alpha=0.5, linewidth=0.5)
# ax.plot(time, y_opt[3, :], color="C0", label="Below")
# ax.plot(time, y_sim[3, :], color="k", linestyle="dotted")
# ax.plot(time, y_opt[6, :], color="C1", label="Above")
# ax.plot(time, y_sim[6, :], color="k", linestyle="dotted")

# rh_out = (
#     params[11]
#     * (d_cl[2, : N + 1] + params[12])
#     / (11 * np.exp(params[26] * d_cl[2, : N + 1] / (d_cl[2, : N + 1] + params[27])))
#     * d_cl[3, : N + 1]
#     * 1e2
# )

# ax.plot(time, rh_out, color="C4", label="Out")
# ax.set_title("Rel. Humidity (\\%)")
# ax.set_ylim(50, 108)
# ax.legend(loc=9, ncol=2)

# ax = axs[1, 0]
# ax.plot(time, d_cl[0, : N + 1] / 1e3, color="C4")
# ax.set_title("Irradiance (W/m$^2$)")
# ax.set_xlabel("t")

# ax = axs[1, 1]
# ax.plot(time[:-1], u_opt[0, :], drawstyle="steps-post", color="C2")
# ax.set_title("CO$_2$ Supply (mg/m$^2$/s)")
# ax.set_xlabel("t")

# ax = axs[1, 2]
# ax.plot(time[:-1], u_opt[2, :], drawstyle="steps-post", color="C2")
# ax.set_title("Heat (W/m$^2$)")
# ax.set_xlabel("t")

# ax = axs[1, 3]
# ax.plot(time[:-1], u_opt[1, :], drawstyle="steps-post", color="C2")
# ax.set_title("Vent (L/m$^2$/s)")
# ax.set_xlabel("t")

# plt.show()

#%%
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
t0 = 0
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

axs[0, 0].plot(time, y_opt[0, :], color="C0", label="Optimal")
axs[0, 0].plot(time, y_sim[0, :], color="k", linestyle="dotted")
axs[0, 0].set_title("Dry Weight [g/m^2]")
beautify(axs[0, 0])

axs[0, 1].axhline(1600, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 1].plot(time, y_opt[1, :] * 1e3, color="C0", label="Below")
axs[0, 1].plot(time, y_sim[1, :] * 1e3, color="k", linestyle="dotted")
axs[0, 1].plot(time, y_opt[4, :] * 1e3, color="C1", label="Above")
axs[0, 1].plot(time, y_sim[4, :] * 1e3, color="k", linestyle="dotted")
axs[0, 1].set_title("CO$_2$ Concentration [ppm]")
axs[0, 1].set_ylim(300, 2000)
axs[0, 1].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 1])

axs[0, 2].axhline(25, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 2].axhline(10, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 2].plot(time, y_opt[2, :], color="C0", label="Below")
axs[0, 2].plot(time, y_sim[2, :], color="k", linestyle="dotted")
axs[0, 2].plot(time, y_opt[5, :], color="C1", label="Above")
axs[0, 2].plot(time, y_sim[5, :], color="k", linestyle="dotted")
axs[0, 2].plot(time, d_cl[2, : N + 1], color="C4", label="Out")
axs[0, 2].set_title("Temperature [^oC]")
axs[0, 2].set_ylim(0, 32)
axs[0, 2].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 2])

rh_out = (
    params[11]
    * (d_cl[2, : N + 1] + params[12])
    / (11 * np.exp(params[26] * d_cl[2, : N + 1] / (d_cl[2, : N + 1] + params[27])))
    * d_cl[3, : N + 1]
    * 1e2
)
axs[0, 3].axhline(80, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 3].plot(time, y_opt[3, :], color="C0", label="Below")
axs[0, 3].plot(time, y_sim[3, :], color="k", linestyle="dotted")
axs[0, 3].plot(time, y_opt[6, :], color="C1", label="Above")
axs[0, 3].plot(time, y_sim[6, :], color="k", linestyle="dotted")
axs[0, 3].plot(time, rh_out, color="C4", label="Out")
axs[0, 3].set_ylim(50, 108)
axs[0, 3].set_title("Relative Humidity [%]")
axs[0, 3].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 3])

# Add a global legend specifically in axes[1, 3]
handles = [
    plt.Line2D([], [], color='C0', linewidth=2, label='Below'),
    plt.Line2D([], [], color='C1', linewidth=2, label="Above"), 
    plt.Line2D([], [], color='C4', linewidth=2, label="Out"),
    plt.Line2D([], [], color='k', linestyle="dotted" , label="Ground Truth"),
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

axs[1, 4].plot(time[:-1], alpha*np.ones(time[:-1].shape), drawstyle="steps-post", color="C4")
axs[1, 4].set_title(r"Flow ($\alpha$) [flow]")
axs[1, 4].set_xlabel("Time (t)")
beautify(axs[1, 4])
# axs[1, 4].axis('off')  # Disable the plot area of axes[1, 3]

# fig.suptitle(f"Integration method: {integration_method}", fontsize=font_size + 3, fontweight='bold')

# === FINAL LAYOUT ===
plt.tight_layout(pad=1.5)
plt.subplots_adjust(wspace=0.5)
plt.show()
