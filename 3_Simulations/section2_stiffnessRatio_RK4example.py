import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory of ../2_Models to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_Models')))
from RK4_integration import RK4
from collocation import collocation

from scipy.linalg import expm, inv

import casadi as ca

#%% Simulate the System with Casadi 
# --- Define System Function (x' = f(t, x, u)) ---
def stiff_dynamics():
    # Create dictionaries 
    p = {}
    x = {}
    u = {}
    d = {}
    # a = {}
    dxdt = {}
    
    # Model parameters (Their values are from the greenlight model)
    # --- 1. Parameters ---
    p['16'] = 3e4
    p['17'] = 1290
    p['18'] = 6.1

    p['leak'] = p['18']
    p['conv'] = p['17']
    p['top'] = p['16'] / 2  # 15000.0
    p['air'] = p['16'] / 2  # 15000.0
    
    x['tAir']      = ca.SX.sym('x_tAir'); 
    x['tTop']      = ca.SX.sym('x_tTop');
    u['thScr']    = ca.SX.sym('u_thScr'); 
    d['tOut']     = ca.SX.sym('d_tOut');

    dxdt['Top'] = 1/p['top']*(-(p['leak'] + p['conv'] * u['thScr'])*x['tTop'] + (p['conv'] * u['thScr'])*x['tAir'] + p['leak']*d['tOut'])
    dxdt['Air'] = 1/p['air']*((p['conv'] * u['thScr'])*x['tTop']  - (p['conv'] * u['thScr'])*x['tAir'])
    
        
    # Stack states and dynamics
    x_vec = ca.vertcat(x['tTop'], x['tAir'])
    u_vec = ca.vertcat(u['thScr'])
    d_vec = ca.vertcat(d['tOut'])
    dx_vec = ca.vertcat(dxdt['Top'], dxdt['Air'])
    
    # Continues time dynamics
    f = ca.Function('f', [x_vec, u_vec, d_vec], [dx_vec])
    
    return f, x_vec, u_vec, d_vec, dx_vec

# --- Load Symbolic Model ---
# Continuous time dynamics
f, x_vec, u_vec, d_vec, dx_vec = stiff_dynamics()

# --- Simulation SetUp ---
nx = 2 # number of states
nu = 1 # number of inputs
nd = 1 # number of disturbances
h = 180 # Step size (seconds)

# Number of simulation steps
N = 80  # Number of simulation steps

x_sim_amin = np.zeros((N, nx))
x_sim_amin[0, :] = [0.0, 0.0] # Initial state x(0) = [0, 0]
x_sim_amax = np.zeros((N, nx))
x_sim_amax[0, :] = [0.0, 0.0] # Initial state x(0) = [0, 0]

u_sim_min = 2.4e-4*np.ones(N)
u_sim_max = 1e-1*np.ones(N)
d_sim = -1*np.ones(N)

# --- Select Integrator ---
integration_method = "RK4"

if integration_method == "collocation":
    F = collocation(h, nx, nu, nd, f)
    
elif integration_method == "RK4":
    x = ca.SX.sym('x',nx)
    u = ca.SX.sym('u',nu)
    d = ca.SX.sym('d',nd)        
    F = RK4(h, x, u, d, f)
    
elif integration_method == "cvodes":
    t0 = 0    
    opts = {'t0':t0, 'tf':h, 'linear_multistep_method':'bdf'}#, 'newton_scheme','bcgstab');
    # Create integrator using CVODES
    ode = {'x':x_vec, 'p': ca.vertcat(u_vec, d_vec), 'ode':dx_vec}
    F = ca.integrator('F', 'cvodes', ode, opts)
    
# ---  Simulation Casadi ---
# Simulate step by step
for k in range(N - 1):
    # Current values
    xk_min = x_sim_amin[k, :]
    xk_max = x_sim_amax[k, :]
    uk_min = u_sim_min[k]
    uk_max = u_sim_max[k]
    dk = d_sim[k]

    # Call the integrator: F(x, u, d) -> x_next
    x_next_amin = F(x0=xk_min, p= ca.vertcat(uk_min, dk))['xf'].full().ravel()
    x_next_amax = F(x0=xk_max, p= ca.vertcat(uk_max, dk))['xf'].full().ravel()
    
    # CasADi returns a DM object, convert to numpy array
    x_sim_amin[k + 1, :] = x_next_amin
    x_sim_amax[k + 1, :] = x_next_amax
    
#%% Calculate the Stiffness Ratio 

def calculate_stiffness_ratio(a, p_leak, p_conv, c_air, c_top):
    """
    Calculates the eigenvalues of matrix A and the stiffness ratio.
    Stiffness Ratio (S) = |lambda_fast| / |lambda_slow|
    """
    # Construct the matrix A
    A = np.array([
        [-(p_leak + p_conv * a) / c_top, (p_conv * a) / c_top],
        [(p_conv * a) / c_air, -(p_conv * a) / c_air]
    ])

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(A)

    # Sort eigenvalues by absolute value (from smallest to largest)
    sorted_eigenvalues = sorted(eigenvalues, key=lambda x: np.abs(x))

    lambda_slow = sorted_eigenvalues[0] # Smallest absolute value
    lambda_fast = sorted_eigenvalues[1] # Largest absolute value

    # Stiffness ratio
    ratio = np.abs(lambda_fast) / np.abs(lambda_slow)

    return ratio

p16 = 3e4
p17 = 1290
p18 = 6.1

p_leak = p18
p_conv = p17
c_top = p16 / 2  # 15000.0
c_air = p16 / 2  # 15000.0
a_min = 2.4e-4
a_max = 1e-1
# Generate 100 points in a logarithmic scale for 'a'
a_values = np.logspace(np.log10(a_min), np.log10(a_max), 100)

# --- Calculate Stiffness Ratios ---
stiffness_ratios = []
for a_val in a_values:
    ratio = calculate_stiffness_ratio(a_val, p_leak, p_conv, c_air, c_top)
    stiffness_ratios.append(ratio)

stiffness_ratios = np.array(stiffness_ratios)

#%% Simulate the System Analytically
x_sim_exp = {}
x_sim_exp['a_min'] = np.zeros((N, nx))
x_sim_exp['a_min'][0, :] = [0.0, 0.0] # Initial state x(0) = [0, 0]
x_sim_exp['a_max'] = np.zeros((N, nx))
x_sim_exp['a_max'][0, :] = [0.0, 0.0] # Initial state x(0) = [0, 0]

for a in [a_min, a_max]:
    # 2x2 System Matrix A
    A = np.array([
        [-(p_leak + p_conv * a) / c_top, (p_conv * a) / c_top],
        [(p_conv * a) / c_air, -(p_conv * a) / c_air]
    ])
    # Disturbance input matrix D_d (2x1)
    D_d = np.array([[p_leak/c_air],[0]])
    # Scalar constant disturbance d
    d = -1.0
    # Initial condition x(0)
    x0 = np.array([0.0, 0.0])
    # Simulation time
    T_end = N*h
    num_points = N
    t = np.linspace(0, T_end, num_points)
    # Calculate the constant disturbance vector v = D_d * d
    v = D_d @ np.array([[d]])
    # Ensure v is a 1D array for easier manipulation (shape (2,))
    v = v.flatten()

    # --- Analytical Solution (using Matrix Exponential) ---
    # Check if A is invertible to use the simplified formula
    if np.linalg.det(A) == 0:
        print("Warning: Matrix A is singular. The simplified formula A^{-1} (e^{At} - I) v will not work.")
        print("A more general approach (e.g., using Laplace transforms or a modified exponential) is needed.")
        # For simplicity, we'll proceed assuming it's usually invertible
        # In a real-world non-invertible scenario, a different approach is necessary
    else:
        A_inv = inv(A)
        # The term (e^{At} - I) * v is calculated as A_inv @ (expm(A*t) - I) @ v        
        x_analytical = np.zeros((num_points, 2))
        for i, t_val in enumerate(t):
            # Calculate the matrix exponential e^{A*t}
            eAt = expm(A * t_val)            
            # Calculate the state x(t) = e^{At} x0 + A^{-1} (e^{At} - I) v
            # x_0 term
            term1 = eAt @ x0            
            # Integral term
            term2 = A_inv @ (eAt - np.identity(2)) @ v            
            x_analytical[i, :] = term1 + term2
        
        if a == a_min:
            x_sim_exp['a_min'] = x_analytical
        else:                
            x_sim_exp['a_max'] = x_analytical

    
# --- 5. Plotting ---
font_size = 20
axis_width = 1.25
# figsize = (8, 10)
figsize = (14, 10)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMU Serif",
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "legend.fontsize": font_size - 6,
    "xtick.labelsize": font_size - 4,
    "ytick.labelsize": font_size - 4,
    "axes.linewidth": axis_width
})


lineWidth = 2
gt_lineWidth = 1.25
d_lineWidth = 1.25
bd_lineWidth = 0.8

# === CREATE SUBPLOTS ===
fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, dpi=300)
# plt.figure(figsize=(10, 6), dpi=300)
time = np.arange(0, 0 + N)*h

def beautify(ax): 
    ax.grid(True, which="both", ls="--")
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)

axs[0].plot(time, d_sim[:], color="k", linewidth = bd_lineWidth, label="$d_T$")
axs[0].plot(time, x_sim_amin[:, 0], color="yellowgreen", marker='o', markersize=4, linestyle='-', label="RK4 - $(a_{min})$")
axs[0].plot(time, x_sim_amax[:, 0], color="darkorchid", marker='o', markersize=4, linestyle='-', label="RK4 - $(a_{max})$")
axs[0].plot(time, x_sim_exp['a_min'][:,0], color="k", markersize=3, linestyle='--', linewidth = gt_lineWidth, label="VoC - $(a_{min})$")
axs[0].plot(time, x_sim_exp['a_max'][:,0], color="k", markersize=3, linestyle='-.', linewidth = gt_lineWidth, label="VoC - $(a_{max})$")
axs[0].set_ylabel("Temperature $[^o\mathrm{C}]$")
# axs[0].set_ylim([-0.5, 1.1])
axs[0].set_ylim([-1.1, 1.1])
axs[0].set_xlim([0, N*h])
axs[0].legend(loc="best", ncol=2, frameon=True, facecolor='white')#, edgecolor='k')
axs[0].set_title('Response of State $x_1(t)$')
beautify(axs[0])

axs[1].plot(time, d_sim[:], color="k", linewidth = bd_lineWidth, label="$d_T$")
axs[1].plot(time, x_sim_amin[:, 1], color="yellowgreen", marker='o', markersize=4, linestyle = '-', linewidth = lineWidth, label="RK4 - $(a_{min})$")
axs[1].plot(time, x_sim_amax[:, 1], color="darkorchid", marker='o', markersize=4, linestyle = '-', linewidth = lineWidth, label="RK4 - $(a_{max})$")
axs[1].plot(time, x_sim_exp['a_min'][:,1], color="k", markersize=3, linestyle='--', linewidth = gt_lineWidth, label="VoC - $(a_{min})$")
axs[1].plot(time, x_sim_exp['a_max'][:,1], color="k", markersize=3, linestyle='-.', linewidth = gt_lineWidth, label="VoC - $(a_{max})$")
axs[1].set_ylabel("Temperature $[^o\mathrm{C}]$")
axs[1].set_xlabel("Time [sec]")
# axs[1].set_ylim([-0.1, 1.5])
axs[1].set_ylim([-1.1, 0.25])
axs[1].set_xlim([0, N*h])
axs[1].legend(loc="best", ncol=2, frameon=True, facecolor='white')#, edgecolor='k')
axs[1].set_title('Response of State $x_2(t)$')
beautify(axs[1])
# fig.suptitle(f"Integration method: {integration_method}", fontsize=font_size + 3, fontweight='bold')
# filename = f'../5_PostSimAnalysis/Section2_StiffnessRatio_RK4Example.pdf'
filename = f'../5_PostSimAnalysis/Section2_RK4Example.pdf'
plt.savefig(filename, dpi=300)  # Save as a PNG with high resolution

# === FINAL LAYOUT ===
plt.tight_layout(pad=1.5)
plt.subplots_adjust(wspace=0.5)
plt.show()

#%% Plot only state 1
plt.figure(figsize=(8, 4), dpi=300)
plt.plot(time, d_sim[:], color="k", linewidth = bd_lineWidth, label="$d_T$")
plt.plot(time, x_sim_amin[:, 1], color="yellowgreen", marker='o', markersize=4, linestyle='-', label="RK4 - $(a_{min})$")
plt.plot(time, x_sim_amax[:, 1], color="darkorchid", marker='o', markersize=4, linestyle='-', label="RK4 - $(a_{max})$")
plt.plot(time, x_sim_exp['a_min'][:,1], color="k", markersize=3, linestyle='--', linewidth = gt_lineWidth, label="Analytic - $(a_{min})$")
plt.plot(time, x_sim_exp['a_max'][:,1], color="k", markersize=3, linestyle='-.', linewidth = gt_lineWidth, label="Analytic - $(a_{max})$")
plt.ylabel("Temperature $[^o\mathrm{C}]$")
# plt.ylim([-0.5, 1.1])
plt.ylim([-1.1, 0.25])
plt.xlim([0, N*h])
plt.legend(loc="best", ncol=2, frameon=True, facecolor='white')#, edgecolor='k')
plt.title('Response of State $x_2(t)$')
plt.xlabel("Time [sec]")
plt.grid(True, which="both", ls="--")


# === FINAL LAYOUT ===
plt.tight_layout(pad=0.2)

filename = f'../5_PostSimAnalysis/Section2_Response_x1.pdf'
plt.savefig(filename,format='pdf', dpi=300)  # Save as a PNG with high resolution
# plt.subplots_adjust(wspace=0.5)
plt.show()


#%% Plot Stiffness Ratio
plt.figure(figsize=(12, 5), dpi=300)
time = np.arange(0, 0 + N)*h
# Plotting with log-log scale
plt.loglog(a_values, stiffness_ratios, marker='o', markersize=3, linestyle='-', label='Stiffness Ratio $\mathcal{S}$')
plt.ylim([0, 1e2])
plt.title('Stiffness Ratio $\mathcal{S}(a)$')
plt.xlabel('$a$')
plt.ylabel('$\mathcal{S}(a)$')
plt.grid(True, which="both", ls="--")
# axs[0].legend(loc="best", ncol=2, frameon=False)

filename = f'../5_PostSimAnalysis/Section2_StiffnessRatio.pdf'
plt.savefig(filename, dpi=300)  # Save as a PNG with high resolution

# === FINAL LAYOUT ===
plt.tight_layout(pad=1.5)
plt.subplots_adjust(wspace=0.5)
plt.show()



