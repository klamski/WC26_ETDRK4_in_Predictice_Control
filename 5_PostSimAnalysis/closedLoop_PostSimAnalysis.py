import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd
import ast
from aux_functions import *

storage = True

# Load Simulated Experiments 
integration_methods = ["ETDRK4", "RK4", "Collocation"]#, "Collocation", "CVODES"]
u_cl = {}
y_cl = {}
d_ol = {}
simDetails = {}
exec_times = {}
solver_exit_msgs = {}
historical = {}

for method in integration_methods:
    u_cl[method] = pd.read_csv(f'../4_OutputData/closedLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_clInputs.csv')
    y_cl[method] = pd.read_csv(f'../4_OutputData/closedLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_clOutputs.csv')
    d_ol[method] = pd.read_csv(f'../4_OutputData/closedLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_weather.csv')
    simDetails[method] = pd.read_csv(f'../4_OutputData/closedLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_simDetails.csv')
    historical[method] = np.load(f'../4_OutputData/closedLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_optimal_groundtruth_predictions.npz')

    # Convert string to list of strings
    solver_exit_msgs[method] = ast.literal_eval(simDetails[method]['Solvers EXIT message'][0])
    # Convert the string with execution times to an array of execution times
    cleaned_string = simDetails[method]['Execution Time [sec]'][0].replace('\n', ' ').replace('\xa0', ' ').strip('[] ')
    time_array_1d = np.fromstring(cleaned_string, sep=' ')
    exec_times[method] = time_array_1d.reshape(-1, 1)
    

#%% Ensure that the loaded experiments are done for the same integration time-intervals and prediction horizon
import pandas as pd

# Define the key DataFrames for comparison
keys_to_check = integration_methods
column_name_1 = 'Integration Timestep [min]'
column_name_2 = 'Prediction Horizon [hours]'
column_name_3 = 'Simulated CL Timesteps'

# --- Execution ---
try:
    check_timesteps_are_equal(simDetails, keys_to_check, column_name_1)
    check_timesteps_are_equal(simDetails, keys_to_check, column_name_2)
    check_timesteps_are_equal(simDetails, keys_to_check, column_name_3)
except (ValueError, KeyError) as e:
    print(f"\n--- Program Halted ---")
    print(e)
    # The program stops here due to the raised exception

#%%
# Generate Plots demonstrating the Closed Loop Behavior Each Integration Method 

Nsim = simDetails['ETDRK4']['Simulated CL Timesteps'].iloc[0] # simDetails[method]['Prediction Horizon [hours]']*60/simDetails[method]['Integration Timestep [min]']
h = int(simDetails[method]['Integration Timestep [min]'].iloc[0].item()*60)


lineWidth = 2
gt_lineWidth = 1.75
d_lineWidth = 1.25
bd_lineWidth = 0.8

for method in integration_methods:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "CMU Serif"
    })
    
    # Create a 3x4 grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), dpi=300)  # 3 rows, 4 columns
    
    # Generate some example data
    x = np.linspace(0, Nsim, Nsim+1)*h/(60*60*24)
    # x = np.linspace(start, end-1, Nsim+1)*h/(60*60*24)
    # x = np.linspace(0, Nsim-1, Nsim)
    
    # Plot States 
    ax = axes[0, 0]    
    ax.plot(x, y_cl[method]['Dry Weight[g/m^2]'], color='orchid', linewidth=gt_lineWidth, linestyle='-')  
    ax.set_title('Dry weight', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{dw}} \ \ [\mathrm{g}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
    # ax.legend(loc='best') # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[0, 1]
    ax.axhline(1600, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.plot(x, y_cl[method]['CO2 concentration[ppm]']* 1e3, color='orchid', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_cl[method]['Top CO2 concentration[ppm]']* 1e3, color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.set_title('CO$_{2}$ Concentration', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{CO}_{2}} \ \ [\mathrm{ppm}]$', fontsize = 14.0)  
    # ax.legend(loc='best', fontsize=12.0) # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[0, 2]
    ax.axhline(25, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.axhline(10, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.plot(x, d_ol[method]['Temperature[^oC]'], color='black', linewidth=d_lineWidth)  
    ax.plot(x, y_cl[method]['Top Air Temperature[^oC]'], color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_cl[method]['Air Temperature[^oC]'], color='orchid', linewidth=lineWidth, linestyle='-')
    ax.set_title('Air temperature', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
    # ax.legend(loc='best', fontsize = 12.0) # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[0, 3]
    ax.axhline(80, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.plot(x, d_ol[method]['Relative Humidity[%]'], color='black', linewidth=d_lineWidth)
    ax.plot(x, y_cl[method]['Relative Humidity[%]'], color='orchid', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_cl[method]['Top Relative Humidity[%]'], color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.set_title('Relative Humidity', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{RH}} \ \ [\%]$', fontsize = 14.0)  # Display ylabel
    # ax.legend(loc='best', fontsize = 12.0) # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    # Add a global legend specifically in axes[0, 4]
    handles = [
        plt.Line2D([], [], color='darkorange', linewidth=lineWidth, label='Top Compartment'),
        plt.Line2D([], [], color='orchid', linewidth=lineWidth, linestyle='-', label='Bottom Compartment'),
        plt.Line2D([], [], color='black', linewidth=d_lineWidth, linestyle='-', label='Outdoor'),
        plt.Line2D([], [], color='darkgreen', linewidth=lineWidth, linestyle='-', label='Control Inputs'),
        plt.Line2D([], [], color='black', linewidth=gt_lineWidth, linestyle=':', label='Ground Truth'),
        plt.Line2D([], [], color='black', alpha=0.5, linewidth=bd_lineWidth, linestyle='--', label='Limits'),        
    ]
    
    # Turn off the axes[1, 3] subplot and use it for the legend
    axes[0, 4].axis('off')  # Disable the plot area of axes[1, 3]
    axes[0, 4].legend(handles=handles, loc='center', fontsize=12, frameon=False)  # Add legend to the center
    
    # Plot weather conditions
    ax = axes[1, 0]
    ax.plot(x, d_ol[method]['Irradiance[W· m^{−2}]'], color='black', linewidth=d_lineWidth)  # Example plot
    ax.set_title('Incoming radiation', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$d_{\mathrm{I}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display legend
    ax.grid(True)  # Add grid
    ax.set_xlabel('Time [Days]', fontsize = 14.0)

    # Plot control inputs
    ax = axes[1, 1]
    ax.plot(x[:Nsim], u_cl[method]['CO2 Supply'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('CO$_{2}$ supply rate ', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{CO}_{2}} \ \ [\mathrm{mg}\!\cdot\!\mathrm{m}^{-2}\!\cdot\!\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 2]
    ax.plot(x[:Nsim], u_cl[method]['Ventilation'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Ventilation rate', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{vent}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 3]
    ax.plot(x[:Nsim], u_cl[method]['Heating'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Heating energy supply', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{heat}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 4]
    ax.plot(x[:Nsim], u_cl[method]['Air Exchange Flow'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Air Exchange Flow', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{flow}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}] \ ??????????$ ', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if storage:
        # Save the figure with the unique identifier in the filename
        filename = f'../5_PostSimAnalysis/closedLoop-ENMPC/{method}_Simulation_Results.pdf'
        plt.savefig(filename)#, dpi=300)  # Save as a PNG with high resolution
    
    # Show the plot
    plt.show()
    
#%%
sDigits = 3
# Mean Execution time for the closed Loop simulated experiment
table = [[' '] + integration_methods]
res = ['t [sec]']
for method in integration_methods:
    res.append(np.round(np.mean(exec_times[method]),sDigits))
table.append(res)    
    
h_val = simDetails[integration_methods[0]]['Integration Timestep [min]'].loc[0]
N_val = simDetails[integration_methods[0]]['Prediction Horizon [hours]'].loc[0]
print_terminal_table(
    data=table, #exectimes_data,
    title=f"Single Open Loop Iteration (h = {h_val}[min], N = {N_val}[hours])"
)

#%% Calculate Economic Performance 
c_co2 = 1e-6 * 0.42 * h
# per kg/s of CO2
c_q = 6.35e-9 * h
# per W of heat
c_dw = -16

 
gt_method = 'ETDRK4'
J_gt  = sum(c_dw * (y_cl[gt_method]['Dry Weight[g/m^2]'].loc[Nsim]- y_cl[gt_method]['Dry Weight[g/m^2]'].loc[0])*1e-3 + c_q * u_cl[gt_method]['Heating'] + c_co2 * u_cl[gt_method]['CO2 Supply'])

labels =  [" ", "Achieved J(x_{CVODES}, u_{method}^{*})", "Abs. Diff from GT", "Rel. Diff from GT"]
Economics = [labels]
for method in integration_methods:
    res = [f'{method}']    
    J_ach = sum(c_dw * (y_cl[method]['Dry Weight[g/m^2]'].loc[Nsim]- y_cl[method]['Dry Weight[g/m^2]'].loc[0])*1e-3 + c_q * u_cl[method]['Heating'] + c_co2 * u_cl[method]['CO2 Supply'])
    abs_dif = J_ach - J_gt
    rel_dif = (J_ach - J_gt)/J_gt*1e2
    res.append(np.round(J_ach,sDigits))
    res.append(np.round(abs_dif,sDigits))
    res.append(np.round(rel_dif,sDigits))
    Economics.append(res)
    
print_terminal_table(
    data=Economics,
    title="Economic Objective Function"
)

#%% Calculate RMSE between predicted and ground truth states 
state_labels = ['Dry Weight[g/m^2]', 'CO2 concentration[ppm]', 'Air Temperature[^oC]', 'Relative Humidity[%]', 'Top CO2 concentration[ppm]', 'Top Air Temperature[^oC]', 'Top Relative Humidity[%]']
RMSE = [[' '] + state_labels]
RMSEperIter = []
for method in integration_methods:
    # RMSEperIter = [f'{method}']
    for i in range(Nsim):
        RMSEperIter.append(np.sqrt(np.mean(np.square(historical[method]['historical_y_opt'][i] - historical[method]['historical_y_gt'][i]),axis=1)))
    RMSE.append([f'{method}'] + np.round(np.mean(RMSEperIter,axis=0),sDigits).tolist())
    
print_terminal_table(
    data=RMSE,
    title=f"RMSE - Predicted vs Ground Truth states"
)


#%% Constraint Violations

constraint = {}
size = int(N_val * 60/h_val)+1
constraint['CO2 concentration[ppm]'] = pd.DataFrame({'min': 0*np.ones(size),'max': 1.6*np.ones(size)})
constraint['Air Temperature[^oC]'] = pd.DataFrame({'min': 10*np.ones(size),'max': 25*np.ones(size)})
constraint['Relative Humidity[%]'] = pd.DataFrame({'min': 0*np.ones(size),'max': 80*np.ones(size)})
constraint['Top CO2 concentration[ppm]'] = pd.DataFrame({'min': 0*np.ones(size),'max': 2*np.ones(size)})
constraint['Top Air Temperature[^oC]'] = pd.DataFrame({'min': 0*np.ones(size),'max': 40*np.ones(size)})
constraint['Top Relative Humidity[%]']  = pd.DataFrame({'min': 0*np.ones(size),'max': 100*np.ones(size)})

state_labels = ['CO2 concentration[ppm]', 'Air Temperature[^oC]', 'Relative Humidity[%]', 'Top CO2 concentration[ppm]', 'Top Air Temperature[^oC]', 'Top Relative Humidity[%]']
CCV = [[' '] + state_labels] # CCV cumulative constraint violation
for method in integration_methods:
    res = [f'{method}']
    for state in state_labels:
        # res.append(np.round(calculate_ccv(y_cl[method][state], constraint[state]['min'], constraint[state]['max'], h_val*60),sDigits))
        res.append(np.round(calculate_ccv(y_cl[method][state], constraint[state]['min'], constraint[state]['max'], 1),sDigits))
    CCV.append(res)
    
print_terminal_table(
    data=CCV,
    title=f"Cumulative Constraint Violation per Climate State"
)