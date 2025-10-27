import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas as pd
from aux_functions import *

storage = True


# Load Simulated Experiments 
integration_methods = ["ETDRK4", "RK4", "Collocation", "CVODES"]
u_opt = {}
y_opt = {}
d_ol = {}
simDetails = {}
y_gt  = {}

for method in integration_methods:
    u_opt[method] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_optInputs.csv')
    y_opt[method] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_optOutputs.csv')
    d_ol[method] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_weather.csv')
    simDetails[method] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_simDetails.csv')
    y_gt[method] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/{method}_ExtvanHentenModel_ENMPC_GroundTruthOutputs.csv')

# u_opt['CVODES'] = pd.read_csv(f'../4_OutputData/openLoop-ENMPC/CVODES_ExtvanHentenModel_ENMPC_optInputs.csv')

#%% Ensure that the loaded experiments are done for the same integration time-intervals and prediction horizon
import pandas as pd

# Define the key DataFrames for comparison
keys_to_check = integration_methods
column_name_1 = 'Integration Timestep [min]'
column_name_2 = 'Prediction Horizon [hours]'


# --- Execution ---
try:
    check_timesteps_are_equal(simDetails, keys_to_check, column_name_1)
    check_timesteps_are_equal(simDetails, keys_to_check, column_name_2)
except (ValueError, KeyError) as e:
    print(f"\n--- Program Halted ---")
    print(e)
    # The program stops here due to the raised exception
    
#%%
# Generate Plots demonstrating the Open Loop Behavior Each Integration Method 

Nsim =  simDetails[method]['Prediction Horizon [hours]']*60/simDetails[method]['Integration Timestep [min]']
h = int(simDetails[method]['Integration Timestep [min]'].iloc[0].item()*60)
Nsim = int(Nsim.iloc[0].item())

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
    ax.plot(x, y_opt[method]['Dry Weight[g/m^2]'], color='orchid', linewidth=gt_lineWidth, linestyle='-')  
    ax.plot(x, y_gt[method]['Dry Weight[g/m^2]'], color='black', linestyle=':', linewidth=lineWidth) 
    ax.set_title('Dry weight', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{dw}} \ \ [\mathrm{g}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
    # ax.legend(loc='best') # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[0, 1]
    ax.axhline(1600, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.plot(x, y_opt[method]['CO2 concentration[ppm]']* 1e3, color='orchid', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_opt[method]['Top CO2 concentration[ppm]']* 1e3, color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_gt[method]['CO2 concentration[ppm]']* 1e3, color='black', linewidth=gt_lineWidth, linestyle=':')
    ax.plot(x, y_gt[method]['Top CO2 concentration[ppm]']* 1e3, color='black', linewidth=gt_lineWidth, linestyle=':')   
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
    ax.plot(x, y_opt[method]['Top Air Temperature[^oC]'], color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_opt[method]['Air Temperature[^oC]'], color='orchid', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_gt[method]['Air Temperature[^oC]'], color='black',  linewidth=gt_lineWidth, linestyle=':')
    ax.plot(x, y_gt[method]['Top Air Temperature[^oC]'], color='black', linewidth=gt_lineWidth, linestyle=':')
    ax.set_title('Air temperature', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$y_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
    # ax.legend(loc='best', fontsize = 12.0) # Display legend
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    # ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[0, 3]
    ax.axhline(80, color="k", linestyle="--", alpha=0.5, linewidth=bd_lineWidth)
    ax.plot(x, d_ol[method]['Relative Humidity[%]'], color='black', linewidth=d_lineWidth)
    ax.plot(x, y_opt[method]['Relative Humidity[%]'], color='orchid', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_opt[method]['Top Relative Humidity[%]'], color='darkorange', linewidth=lineWidth, linestyle='-')
    ax.plot(x, y_gt[method]['Relative Humidity[%]'], color='black', linewidth=gt_lineWidth, linestyle=':')
    ax.plot(x, y_gt[method]['Top Relative Humidity[%]'], color='black', linewidth=gt_lineWidth, linestyle=':')
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
    ax.plot(x[:Nsim], u_opt[method]['CO2 Supply'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('CO$_{2}$ supply rate ', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{CO}_{2}} \ \ [\mathrm{mg}\!\cdot\!\mathrm{m}^{-2}\!\cdot\!\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 2]
    ax.plot(x[:Nsim], u_opt[method]['Ventilation'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Ventilation rate', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{vent}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 3]
    ax.plot(x[:Nsim], u_opt[method]['Heating'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Heating energy supply', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{heat}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    ax = axes[1, 4]
    ax.plot(x[:Nsim], u_opt[method]['Air Exchange Flow'], color='darkgreen', linewidth=lineWidth, linestyle='-') 
    ax.set_title('Air Exchange Flow', fontsize = 16.0)  # Title for each subplot
    ax.set_ylabel('$u_{\mathrm{flow}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}] \ ??????????$ ', fontsize = 14.0)  # Display ylabel
    ax.grid(True)  # Add grid
    # ax.set_xlim(start, end)
    ax.set_xlabel('Time [Days]', fontsize = 14.0)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if storage:
        # Save the figure with the unique identifier in the filename
        filename = f'../5_PostSimAnalysis/openLoop-ENMPC/{method}_Simulation_Results.pdf'
        plt.savefig(filename)#, dpi=300)  # Save as a PNG with high resolution
    
    # Show the plot
    plt.show()

#%% Plot Absolute Errors Between Predictions and Ground Truth measurements
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMU Serif"
})

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=300)  # 3 rows, 4 columns

fig.suptitle(
   " Abs. Error of Predicted States",
    fontsize=20,
    fontweight='bold',
    color='black',
    # y=1.02  # Adjust vertical position if needed
)
# Generate some example data
x = np.linspace(0, Nsim, Nsim+1)*h/(60*60*24)
# x = np.linspace(start, end-1, Nsim+1)*h/(60*60*24)
# x = np.linspace(0, Nsim-1, Nsim)

# Plot States 
ax = axes[0, 0]    
ax.plot(x, np.abs(y_opt['ETDRK4']['Dry Weight[g/m^2]'] - y_gt['ETDRK4']['Dry Weight[g/m^2]']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Dry Weight[g/m^2]'] - y_gt['RK4']['Dry Weight[g/m^2]']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Dry Weight[g/m^2]'] - y_gt['Collocation']['Dry Weight[g/m^2]']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Dry weight', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{dw}} \ \ [\mathrm{g}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[0, 1]
ax.plot(x, np.abs(y_opt['ETDRK4']['CO2 concentration[ppm]'] - y_gt['ETDRK4']['CO2 concentration[ppm]'])* 1e3, color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['CO2 concentration[ppm]'] - y_gt['RK4']['CO2 concentration[ppm]'])* 1e3, color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['CO2 concentration[ppm]'] - y_gt['Collocation']['CO2 concentration[ppm]'])* 1e3, color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('CO$_{2}$ Concentration', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{CO}_{2}} \ \ [\mathrm{ppm}]$', fontsize = 14.0)  
# ax.legend(loc='best', fontsize=12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[0, 2]
ax.plot(x, np.abs(y_opt['ETDRK4']['Air Temperature[^oC]'] - y_gt['ETDRK4']['Air Temperature[^oC]']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Air Temperature[^oC]'] - y_gt['RK4']['Air Temperature[^oC]']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Air Temperature[^oC]'] - y_gt['Collocation']['Air Temperature[^oC]']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Air temperature', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[0, 3]
ax.plot(x, np.abs(y_opt['ETDRK4']['Relative Humidity[%]'] - y_gt['ETDRK4']['Relative Humidity[%]']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Relative Humidity[%]'] - y_gt['RK4']['Relative Humidity[%]']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Relative Humidity[%]'] - y_gt['Collocation']['Relative Humidity[%]']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Relative Humidity', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{RH}} \ \ [\%]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[1, 0]
ax.plot(x, np.abs(y_opt['ETDRK4']['Top CO2 concentration[ppm]'] - y_gt['ETDRK4']['Top CO2 concentration[ppm]'])* 1e3, color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Top CO2 concentration[ppm]'] - y_gt['RK4']['Top CO2 concentration[ppm]'])* 1e3, color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Top CO2 concentration[ppm]'] - y_gt['Collocation']['Top CO2 concentration[ppm]'])* 1e3, color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('CO$_{2}$ Concentration', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{CO}_{2}} \ \ [\mathrm{ppm}]$', fontsize = 14.0)  
# ax.legend(loc='best', fontsize=12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[1, 1]
ax.plot(x, np.abs(y_opt['ETDRK4']['Top Air Temperature[^oC]'] - y_gt['ETDRK4']['Top Air Temperature[^oC]']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Top Air Temperature[^oC]'] - y_gt['RK4']['Top Air Temperature[^oC]']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Top Air Temperature[^oC]'] - y_gt['Collocation']['Top Air Temperature[^oC]']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Air temperature', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{T}} \ \ [^o\mathrm{C}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[1, 2]
ax.plot(x, np.abs(y_opt['ETDRK4']['Top Relative Humidity[%]'] - y_gt['ETDRK4']['Top Relative Humidity[%]']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['RK4']['Top Relative Humidity[%]'] - y_gt['RK4']['Top Relative Humidity[%]']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x, np.abs(y_opt['Collocation']['Top Relative Humidity[%]'] - y_gt['Collocation']['Top Relative Humidity[%]']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Relative Humidity', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{RH}} \ \ [\%]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

# Add a global legend specifically in axes[0, 4]
handles = [
    plt.Line2D([], [], color='orchid', linewidth=lineWidth, label='ETDRK4-CVODES'),
    plt.Line2D([], [], color='olive', linewidth=lineWidth, linestyle='-', label='RK4-CVODES'),
    plt.Line2D([], [], color='steelblue', linewidth=d_lineWidth, linestyle='-', label='Collocation-CVODES'),
]

# Turn off the axes[1, 3] subplot and use it for the legend
axes[1, 3].axis('off')  # Disable the plot area of axes[1, 3]
axes[1, 3].legend(handles=handles, loc='center', fontsize=12, frameon=False)  # Add legend to the center    


# Adjust layout to prevent overlap
plt.tight_layout()

if storage:
    # Save the figure with the unique identifier in the filename
    filename = f'../5_PostSimAnalysis/openLoop-ENMPC/Measurements_Absolute_Errors.pdf'
    plt.savefig(filename)#, dpi=300)  # Save as a PNG with high resolution

# Show the plot
plt.show()

#%% ||u_method^* - u_CVODES^*||

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMU Serif"
})

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(15, 8), dpi=300)  # 3 rows, 4 columns

fig.suptitle(
   " Abs. Error of Control Inputs",
    fontsize=20,
    fontweight='bold',
    color='black',
    # y=1.02  # Adjust vertical position if needed
)

# Generate some example data
x = np.linspace(0, Nsim, Nsim+1)*h/(60*60*24)
# x = np.linspace(start, end-1, Nsim+1)*h/(60*60*24)
# x = np.linspace(0, Nsim-1, Nsim)

# Plot States 
ax = axes[0]#[0, 0]    
ax.plot(x[:Nsim], np.abs(u_opt['ETDRK4']['CO2 Supply'] - u_opt['CVODES']['CO2 Supply']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['RK4']['CO2 Supply'] - u_opt['CVODES']['CO2 Supply']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['Collocation']['CO2 Supply'] - u_opt['CVODES']['CO2 Supply']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('CO$_{2}$ supply rate ', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{CO}_{2}} \ \ [\mathrm{mg}\!\cdot\!\mathrm{m}^{-2}\!\cdot\!\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best') # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[1]#[0, 1]
ax.plot(x[:Nsim], np.abs(u_opt['ETDRK4']['Ventilation'] - u_opt['CVODES']['Ventilation'])* 1e3, color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['RK4']['Ventilation'] - u_opt['CVODES']['Ventilation'])* 1e3, color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['Collocation']['Ventilation'] - u_opt['CVODES']['Ventilation'])* 1e3, color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Ventilation rate', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{vent}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize=12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[2]#[0, 2]
ax.plot(x[:Nsim], np.abs(u_opt['ETDRK4']['Heating'] - u_opt['CVODES']['Heating']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['RK4']['Heating'] - u_opt['CVODES']['Heating']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['Collocation']['Heating'] - u_opt['CVODES']['Heating']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Heating energy supply', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{heat}} \ \ [\mathrm{W}\!\cdot\!\mathrm{m}^{-2}]$', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

ax = axes[3]#[0, 3]
ax.plot(x[:Nsim], np.abs(u_opt['ETDRK4']['Air Exchange Flow'] - u_opt['CVODES']['Air Exchange Flow']), color='orchid', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['RK4']['Air Exchange Flow'] - u_opt['CVODES']['Air Exchange Flow']), color='olive', linewidth=gt_lineWidth, linestyle='-')  
ax.plot(x[:Nsim], np.abs(u_opt['Collocation']['Air Exchange Flow'] - u_opt['CVODES']['Air Exchange Flow']), color='steelblue', linewidth=gt_lineWidth, linestyle='-')  
ax.set_title('Air Exchange Flow', fontsize = 16.0)  # Title for each subplot
ax.set_ylabel('$|e|_{\mathrm{flow}} \ \ [\mathrm{L}\cdot\mathrm{m}^{-2}\cdot\mathrm{s}^{-1}] \ ??????????$ ', fontsize = 14.0)  # Display ylabel
# ax.legend(loc='best', fontsize = 12.0) # Display legend
ax.grid(True)  # Add grid
# ax.set_xlim(start, end)
# ax.set_xlabel('Time [Days]', fontsize = 14.0)

# Adjust layout to prevent overlap
plt.tight_layout()

if storage:
    # Save the figure with the unique identifier in the filename
    filename = f'../5_PostSimAnalysis/openLoop-ENMPC/ControlInputs_Absolute_Errors.pdf'
    plt.savefig(filename)#, dpi=300)  # Save as a PNG with high resolution

# Show the plot
plt.show()

#%%

# Execution time for a single iteration of the openloop EMPC per method
table = [[' '] + integration_methods]
res = ['t [sec]']
for method in integration_methods:
    res.append(np.round(simDetails[method]['Execution Time [sec]'].loc[0],4))
table.append(res)    
    
h_val = simDetails['RK4']['Integration Timestep [min]'].loc[0]
N_val = simDetails['RK4']['Prediction Horizon [hours]'].loc[0]
print_terminal_table(
    data=table, #exectimes_data,
    title=f"Single Open Loop Iteration (h = {h_val}[min], N = {N_val}[hours])"
)

# Calculate RMSE between predicted and ground truth states 
state_labels = ['Dry Weight[g/m^2]', 'CO2 concentration[ppm]', 'Air Temperature[^oC]', 'Relative Humidity[%]', 'Top CO2 concentration[ppm]', 'Top Air Temperature[^oC]', 'Top Relative Humidity[%]']
RMSE = [[' '] + state_labels]
for method in integration_methods:
    res = [f'{method}']
    for state in state_labels:
        res.append(np.round(np.sqrt(np.mean(np.square(y_opt[method][state] - y_gt[method][state]))),4))
    RMSE.append(res)
    
print_terminal_table(
    data=RMSE,
    title=f"RMSE - Predicted vs Ground Truth states"
)

# Calculate Correlation Coefficient (Pearson's r)
from scipy.stats import pearsonr
state_labels = ['Dry Weight[g/m^2]', 'CO2 concentration[ppm]', 'Air Temperature[^oC]', 'Relative Humidity[%]', 'Top CO2 concentration[ppm]', 'Top Air Temperature[^oC]', 'Top Relative Humidity[%]']
R2 = [[' '] + state_labels]
for method in integration_methods:
    res = [f'{method}']
    for state in state_labels:
        r_x, p_value_x = pearsonr(y_opt[method][state], y_gt[method][state])
        res.append(np.round(r_x,4))
    R2.append(res)
    
print_terminal_table(
    data=R2,
    title=f"Pearson R2 - Predicted vs Ground Truth states"
)

# Calculate RMSE between u_{method}^{*} and u_{CVODES}^{*}
input_labels = ['CO2 Supply', 'Ventilation', 'Heating', 'Air Exchange Flow']
RMSE_input = [[' '] + input_labels]
for method in integration_methods:
    res = [f'{method}']
    for control in input_labels:
        res.append(np.round(np.sqrt(np.mean(np.square(u_opt[method][control] - u_opt['CVODES'][control]))),4))
    RMSE_input.append(res)
    
print_terminal_table(
    data=RMSE_input,
    title="RMSE - $u_{method}^*$ vs $u_{CVODES}^*$"
)

# Calculate Correlation Coefficient (Pearson's r) between u_{method}^{*} and u_{CVODES}^{*}
R2_input = [[' '] + input_labels]
for method in integration_methods:
    res = [f'{method}']
    for control in input_labels:
        r_x, p_value_x = pearsonr(u_opt[method][control], u_opt['CVODES'][control])
        res.append(np.round(r_x,4))
    R2_input.append(res)
    
print_terminal_table(
    data=R2_input,
    title="Pearson R2 - u_{method}^{*} vs u_{CVODES}^{*}"
)
# Calculate Economic Performance 
c_co2 = 1e-6 * 0.42 * h
# per kg/s of CO2
c_q = 6.35e-9 * h
# per W of heat
c_dw = -16

labels =  [" ", "Predicted J(x_{method}, u_{method}^{*})", "Achieved J(x_{CVODES}, u_{method}^{*})", "Abs. Diff", "Rel. Diff"]
Economics = [labels]
for method in integration_methods:
    res = [f'{method}']
    # for label in labels:
    J_pr  = sum(c_dw * (y_opt[method]['Dry Weight[g/m^2]'].loc[Nsim]- y_opt[method]['Dry Weight[g/m^2]'].loc[0])*1e-3 + c_q * u_opt[method]['Heating'] + c_co2 * u_opt[method]['CO2 Supply'])
    J_ach = sum(c_dw * (y_gt[method]['Dry Weight[g/m^2]'].loc[Nsim]- y_gt[method]['Dry Weight[g/m^2]'].loc[0])*1e-3 + c_q * u_opt[method]['Heating'] + c_co2 * u_opt[method]['CO2 Supply'])
    abs_dif = abs(J_pr - J_ach)
    rel_dif = (J_pr - J_ach)/J_ach*1e2
    res.append(np.round(J_pr,4))
    res.append(np.round(J_ach,4))
    res.append(np.round(abs_dif,4))
    res.append(np.round(rel_dif,4))
    Economics.append(res)
    
print_terminal_table(
    data=Economics,
    title="Economic Objective Function"
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
        res.append(np.round(calculate_ccv(y_gt[method][state], constraint[state]['min'], constraint[state]['max'], h_val*60),4))
    CCV.append(res)
    
print_terminal_table(
    data=CCV,
    title=f"Cumulative Constraint Violation per Climate State"
)