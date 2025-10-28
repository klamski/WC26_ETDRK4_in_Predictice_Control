import numpy as np
import pandas as pd
import scipy.io
import scipy.interpolate
import scipy.integrate
import casadi as ca
import matplotlib.pyplot as plt
import sys
import os
import math
from datetime import datetime
import time

# Add the parent directory of ../2_Models to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../2_Models")))
import extended_stiff_vanhenten_model as ca_model
# import vanhenten_model as model
from RK4_integration import RK4
from collocation import collocation
from stiff_dynamics import stiff_dynamics

# Decide if you want to store the results
storage = True
unique_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#%% Open the file in read mode
data = scipy.io.loadmat("../1_InputData/weather.mat")
weather = {k: data[k] for k in ["co2", "hum", "iGlob", "tOut"]}
params = ca_model.params


Delta = 30 * 60  # 30 minutes
N = 2 * 24 * 2  # 2 days with 30 min time steps

# Delta = 1 * 60  # 15 minutes
# N =  60 * 6   

nx = 7
nu = 4 
nd = 4

t0 = 0
# x0 = np.array([0.0035, 0.001, 15, 0.008, 0.002, 20, 0.012])
x0 = np.array([0.0035, 0.001, 15, 0.008, 0.00085, 13, 0.01])

c_co2 = 1e-6 * 0.42 * Delta
# per kg/s of CO2
c_q = 6.35e-9 * Delta
# per W of heat
c_dw = -16
# price per kg of dry weight


d_ol = np.array([weather["iGlob"], weather["co2"], weather["tOut"], weather["hum"]])

d_ol = d_ol[..., 0]


def interpolate_weather(d_ol, Delta):
    original_length = len(d_ol)
    original_indices = np.arange(original_length)

    step = Delta / 300  # E.g., 3.0 for 15-min from 5-min
    original_indices = np.arange(d_ol.shape[1])
    new_indices = np.arange(0, d_ol.shape[1], step)

    # Create an interpolator for each feature (row)
    interp_func = scipy.interpolate.interp1d(
        original_indices, d_ol, axis=1, kind="linear", fill_value="extrapolate"
    )

    # Interpolate
    return interp_func(new_indices)


d_ol = interpolate_weather(d_ol, Delta)

####################### LOAD AND INTEGRATE LETTUCE MODELS ######################
from RK4_integration import RK4
x = ca.MX.sym("x", nx)
u = ca.MX.sym("u", nu)
d = ca.MX.sym("d", nd)

M = ca_model.odes_screen_lin(params)
A = M*u[3]
 
F = ca.Function("F", [x, u, d], [A @ x + ca_model.odes_screen_nonlin(x, u, d, params)], ["x", "u", "d"], ["xplus"])
h = Delta

# Set Up the ground truth model (CVODES)
t0 = 0    
opts = {'t0':t0, 'tf':h, 'linear_multistep_method':'bdf'}#, 'newton_scheme','bcgstab');
# Create integrator using CVODES
ode = {'x':x, 'p': ca.vertcat(u, d), 'ode':A @ x + ca_model.odes_screen_nonlin(x, u, d, params)}
f_gt = ca.integrator('f_gt', 'cvodes', ode, opts)


integration_method = "Collocation"

if integration_method == "Collocation":
    f = collocation(h, nx, nu, nd, F)
    
elif integration_method == "RK4":
    x = ca.SX.sym('x',nx)
    u = ca.SX.sym('u',nu)
    d = ca.SX.sym('d',nd)        
    f = RK4(h, x, u, d, F)
    
elif integration_method == "CVODES":
    t0 = 0    
    opts = {'t0':t0, 'tf':h, 'linear_multistep_method':'bdf'}#, 'newton_scheme','bcgstab'); #CVODES opts
    # opts = {'t0':t0, 'tf':h} #collocation opts
    # Create integrator using CVODES
    ode = {'x':x, 'p': ca.vertcat(u, d), 'ode':A @ x + ca_model.odes_screen_nonlin(x, u, d, params)}
    f = ca.integrator('f', 'cvodes', ode, opts)
    # f = ca.integrator('f', 'rk', ode, opts)
    
elif integration_method == "ETDRK4":
    M = ca_model.odes_screen_lin(params)
    # A(a) = M a where a = c1(1-u3) + c2
    # Define a
    a = ca.MX.sym('a',1)
    # Define V, L, V_inv, eigenvalues
    eigvals, V = scipy.linalg.eigh(M)
    V_inv = scipy.linalg.inv(V)
    # Replace very small eigenvalues with zero
    eigvals[np.abs(eigvals)<1e-6] = 0 
    # Convert to Casadi objects
    V = ca.MX(V)
    V_inv = ca.MX(V_inv)
    def phi_n_scalar_sym(z, a, n, terms=1):
        if abs(z) < 1e-6:
            return ca.MX(sum(z**k / math.factorial(k + n) for k in range(terms)))
        else:
            if n == 1:
                return (ca.exp(z*a) - 1) / (z*a)
            elif n == 2:
                return (ca.exp(z*a) - 1 - z*a) / ((z*a)**2)
            elif n == 3:
                return (ca.exp(z*a) - 1 - z*a - (z*a)**2 / 2) / ((z*a)**3)
         
    phi1 = ca.Function("phi1", [a],  [V @ ca.diag(ca.vertcat(*[phi_n_scalar_sym(lam*Delta, a, 1) for lam in eigvals]))  @ V_inv], ["u"], ["phi_1"])
    phi2 = ca.Function("phi2", [a],  [V @ ca.diag(ca.vertcat(*[phi_n_scalar_sym(lam*Delta, a, 2) for lam in eigvals]))  @ V_inv], ["u"], ["phi_2"])
    phi3 = ca.Function("phi3", [a],  [V @ ca.diag(ca.vertcat(*[phi_n_scalar_sym(lam*Delta, a, 3) for lam in eigvals]))  @ V_inv], ["u"], ["phi_3"])
           
    Q = ca.Function("Q", [a],  [Delta / 2 * V @ ca.diag(ca.vertcat(*[phi_n_scalar_sym(lam*Delta/2, a, 1) for lam in eigvals]))  @ V_inv], ["u"], ["Q_"])

    def etdrk4_sym(V, eigvals, V_inv, phi1, phi2, phi3, Q, f, x0, dt):
        alpha = ca.MX.sym('alpha', 1)
        # E = V e^{L a dt} V^{-1}
        # exp_sym = ca.diag(ca.vertcat(*[ca.exp(lam*alpha*dt) for lam in eigvals]))
        E = ca.Function("E", [alpha],  [V @ ca.diag(ca.vertcat(*[ca.exp(lam*alpha*dt) for lam in eigvals])) @ V_inv], ["u"], ["Exp"])
        
        # E2 = V e^{L a dt/2} V^{-1}
        # exp_sym2 = ca.diag(ca.vertcat(*[ca.exp(lam*alpha*dt/2) for lam in eigvals]))
        E2 = ca.Function("E2", [alpha],  [V @ ca.diag(ca.vertcat(*[ca.exp(lam*alpha*dt/2) for lam in eigvals])) @ V_inv], ["u"], ["Exp2"])

        # f0 = f(x0)
        # a = E2 @ x0 + Q @ f0
        # fa = f(a)
        # b = E2 @ x0 + Q @ fa
        # fb = f(b)
        # c = E2 @ a + Q @ (2 * fb - f0)
        # fc = f(c) 
        f0 = f(x0)
        a = E2(u[3]) @ x0 + Q(u[3]) @ f0
        fa = f(a)
        b = E2(u[3]) @ x0 + Q(u[3]) @ fa
        fb = f(b)
        c = E2(u[3]) @ a + Q(u[3]) @ (2 * fb - f0)
        fc = f(c) 
        
        k1 = phi1(u[3]) - 3 * phi2(u[3]) + 4 * phi3(u[3])
        k2 = 2 * phi2(u[3]) - 4 * phi3(u[3])
        k3 = -1 * phi2(u[3]) + 4 * phi3(u[3])
        
        x = E(u[3]) @ x0 + dt * (k1 @ f0 + k2 @ (fa + fb) + k3 @ fc)
        
        return x

    def f_etd(x, u, d):
        return etdrk4_sym(V, eigvals, V_inv, phi1, phi2, phi3, Q, lambda x: ca_model.odes_screen_nonlin(x, u, d, params), x, Delta)
        
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    d = ca.MX.sym("d", nd)

    f = ca.Function("f", [x, ca.vertcat(u,d)], [f_etd(x, u, d)], ["x0", "p"], ["xf"])
    

# Set up the Output function
h = ca.Function("h", [x], [ca_model.h_meas_screen(x, params)], ["x"], ["y"])

###################### SET UP OPEN-LOOP MPC ###################################
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

d_p = ca.MX.sym("d_p", nd, N)

lbx = [0, 1e-4, 5, 0, 0, 0, 0]
ubx = [600, 0.009, 50, 1, 0.009, 50, 1]
lbu = [0, 0, 0, 2.4e-4]
ubu = [1.2, 7.5, 150, 0.1]

# alpha0 = np.ones(N)*1e-6 #np.zeros(N)
# alpha0[np.nonzero(d_ol[0, : N + 1])] = 0.1

Xk = ca.MX.sym("X_0", nx)
w.append(Xk)
lbw.append(lbx)
ubw.append(ubx)
w0.append(lbx)
x_out.append(Xk)
y_out.append(h(Xk))

for k in range(N):
    Uk = ca.MX.sym("U_" + str(k), nu)
    w.append(Uk)
    lbw.append(lbu)#[0, 0, 0, alpha0[k]])
    ubw.append(ubu)#[1.2, 7.5, 150, alpha0[k]])
    w0.append(lbu)
    u_out.append(Uk)

    Xk_plus = f(x0=Xk, p= ca.vertcat(Uk, d_p[:, k]))["xf"]
    # x_next = F(x0=xk, p= ca.vertcat(uk, dk))['xf'].full().ravel()

    J = J + c_dw * (Xk_plus[0] - Xk[0]) + c_q * Uk[2] + c_co2 * Uk[0]

    Xk = ca.MX.sym("X_" + str(k + 1), nx)
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
        "warm_start_init_point": "yes",
        # 'tol': 1e-6,           # Convergence tolerance
    },
    # "print_time": False,  # Print timing information
    # "verbose": False,  # CasADi-level verbosity
}

w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_out = ca.horzcat(*x_out)
u_out = ca.horzcat(*u_out)
y_out = ca.horzcat(*y_out)

prob = {
    "f": J,
    "x": w,
    "g": g,
    "p": d_p,
}

solver = ca.nlpsol("solver", "ipopt", prob, opts)

trajectories = ca.Function(
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

start_time = time.time()
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=d_ol[:, t0:N])
end_time = time.time()
print(f'Execution time: {end_time - start_time}  [sec]')

x_opt, u_opt, y_opt = trajectories(sol["x"])
x_opt = x_opt.full()
u_opt = u_opt.full()
y_opt = y_opt.full()

#%% #################### SIMULATE GROUND TRUTH ##################################
x_sim = np.zeros((nx,N+1))
xk_plus = x0
x_sim[:, 0] = x0
for k in range(N):    
    xk_plus = f_gt(x0=xk_plus, p= ca.vertcat(u_opt[:,k], d_ol[:, k]))["xf"].full().ravel()
    x_sim[:, k+1] = xk_plus

y_sim = h(x_sim).full()

#%%
import matplotlib.pyplot as plt
import numpy as np

# === CONTROL VARIABLES ===
font_size = 11
axis_width = 1.2
figsize = (12.5, 6)

# === GLOBAL STYLE SETTINGS ===
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Serif",
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
# axs[0, 1].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 1])

axs[0, 2].axhline(30, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 2].axhline(10, color="k", linestyle="--", alpha=0.5, linewidth=0.8)
axs[0, 2].plot(time, y_opt[2, :], color="C0", label="Below")
axs[0, 2].plot(time, y_sim[2, :], color="k", linestyle="dotted")
axs[0, 2].plot(time, y_opt[5, :], color="C1", label="Above")
axs[0, 2].plot(time, y_sim[5, :], color="k", linestyle="dotted")
axs[0, 2].plot(time, d_ol[2, : N + 1], color="C4", label="Out")
axs[0, 2].set_title("Temperature [^oC]")
axs[0, 2].set_ylim(0, 32)
# axs[0, 2].legend(loc="upper center", ncol=2, frameon=False)
beautify(axs[0, 2])

rh_out = (
    params[11]
    * (d_ol[2, : N + 1] + params[12])
    / (11 * np.exp(params[26] * d_ol[2, : N + 1] / (d_ol[2, : N + 1] + params[27])))
    * d_ol[3, : N + 1]
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
# axs[0, 3].legend(loc="upper center", ncol=2, frameon=False)
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
axs[1, 0].plot(time, d_ol[0, : N + 1] / 1e3, color="C4")
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

# for i, (name, unit) in enumerate(labels):s
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

#%% Save as CSV
# Convert to pandas dataFrames before saving
u_labels = ['CO2 Supply', 'Ventilation', 'Heating', 'Air Exchange Flow']
y_labels = ['Dry Weight[g/m^2]', 'CO2 concentration[ppm]', 'Air Temperature[^oC]', 'Relative Humidity[%]', 'Top CO2 concentration[ppm]', 'Top Air Temperature[^oC]', 'Top Relative Humidity[%]']
d_labels = ['Irradiance[W· m^{−2}]', 'CO2[kg·m^{−3}]', 'Temperature[^oC]', 'Humidity[kg·m^{−3}]', 'Relative Humidity[%]']
simDetail_labels = ['Integration Method', 'Integration Timestep [min]', 'Prediction Horizon [hours]', 'Execution Time [sec]', 'Solvers EXIT message']
simDetails = [integration_method, Delta/60, N*Delta/3600, end_time - start_time, solver.stats()["return_status"]]

u_opt_df = pd.DataFrame(u_opt.T, columns=u_labels)
y_opt_df = pd.DataFrame(y_opt.T, columns=y_labels)
d_ol_df = pd.DataFrame(np.hstack((d_ol[:,0:N+1].T, rh_out.reshape(-1, 1))), columns=d_labels)
simDetails_pd = pd.DataFrame([simDetails], columns=simDetail_labels)
y_gt_df = pd.DataFrame(y_sim.T, columns=y_labels)

if storage:
    u_opt_df.to_csv(f'../4_OutputData/openLoop-ENMPC/{integration_method}_ExtvanHentenModel_ENMPC_optInputs.csv', index=False)
    y_opt_df.to_csv(f'../4_OutputData/openLoop-ENMPC/{integration_method}_ExtvanHentenModel_ENMPC_optOutputs.csv', index=False)
    d_ol_df.to_csv(f'../4_OutputData/openLoop-ENMPC/{integration_method}_ExtvanHentenModel_ENMPC_weather.csv', index=False)
    simDetails_pd.to_csv(f'../4_OutputData/openLoop-ENMPC/{integration_method}_ExtvanHentenModel_ENMPC_simDetails.csv', index=False)
    y_gt_df.to_csv(f'../4_OutputData/openLoop-ENMPC/{integration_method}_ExtvanHentenModel_ENMPC_GroundTruthOutputs.csv', index=False)
    
    