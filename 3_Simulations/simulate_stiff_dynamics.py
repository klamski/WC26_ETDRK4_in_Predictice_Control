import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add the parent directory of ../2_Models to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2_Models')))
from RK4_integration import RK4
from collocation import collocation
from stiff_dynamics import stiff_dynamics

################# LOAD RECORDED DATA ##########################################
# load weather data 
d_val = np.loadtxt('../1_InputData/gl_weather_subset.csv',delimiter=',',skiprows=1)
# load input data 
u_val = np.loadtxt('../1_InputData/gl_inputs_thScr_heat.csv',delimiter=',',skiprows=1)
# load state data 
x_val = np.loadtxt('../1_InputData/gl_states_tAir_tTop.csv',delimiter=',',skiprows=1)

################ LOAD SYMOBLIC MODEL ##########################################

# Continuous time dynamics
f, x_vec, u_vec, d_vec, dx_vec = stiff_dynamics()

################# CHOOSE INTEGRATION METHOD ##################################
h = 300
integration_method = "collocation"

if integration_method == "collocation":
    nx = 2
    nu = 2
    nd = 3
    F = collocation(h, nx, nu, nd, f)
    
elif integration_method == "RK4":
    x = ca.SX.sym('x',2)
    u = ca.SX.sym('u',2)
    d = ca.SX.sym('d',3)        
    F = RK4(h, x, u, d, f)
    
elif integration_method == "cvodes":
    t0 = 0    
    opts = {'t0':t0, 'tf':h, 'linear_multistep_method':'bdf'}#, 'newton_scheme','bcgstab');
    # Create integrator using CVODES
    ode = {'x':x_vec, 'p': ca.vertcat(u_vec, d_vec), 'ode':dx_vec}
    F = ca.integrator('F', 'cvodes', ode, opts)
        

############################# SIMULATE ######################################## 
# Number of simulation steps
N = u_val.shape[0]  # Or d_val.shape[0], assuming they're the same length

# Initialize state storage
x_sim = np.zeros((N, x_val.shape[1]))
x_sim[0, :] = x_val[0, :]  # Initial state

# Simulate step by step
for k in range(N - 1):
    # Current values
    xk = x_sim[k, :]
    uk = u_val[k, :]
    dk = d_val[k, :]

    # Call the integrator: F(x, u, d) -> x_next
    x_next = F(x0=xk, p= ca.vertcat(uk, dk))['xf'].full().ravel()
    
    # CasADi returns a DM object, convert to numpy array
    x_sim[k + 1, :] = x_next

# Done! Let's plot the simulated vs actual states
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x_sim[:, 0], label='Simulated tAir', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('tAir (°C)')
# plt.ylim([-20,50])
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_sim[:, 1], label='Simulated tTop', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('tTop (°C)')
# plt.ylim([-20,50])
plt.legend()

plt.tight_layout()
plt.show()

