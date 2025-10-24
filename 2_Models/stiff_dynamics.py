import casadi as ca

def stiff_dynamics():
    # Create dictionaries 
    p = {}
    x = {}
    u = {}
    d = {}
    a = {}
    dxdt = {}
    
    # Model parameters (Their values are from the greenlight model)
    p['rhoAir']     = 1.2   # Density of the air [kg m^{-3}]
    p['cPAir']      = 1e3   # Specific heat capacity of air [J K^{-1} kg^{-1}] 
    # p['hFlr']       = 0.02  # Thickness of floor [m] 
    # p['lambdaFlr']  = 1.7   # Thermal heat conductivity of the floor [m^{-1} K^{-1}]
    p['capAir']     = 7560  # Heat capacity of air [J K^{-1} m^{-2}]
    p['capTop']     = 726   # Heat capacity of air in top compartment [J K^{-1} m^{-2}]
    p['cLeakage']   =  1e-4 # Greenhouse leakage coefficien	[-]
    p['hSo1']       = 0.04  # Thickness of soil layer 1 [m]
    p['lambdaSo']   = 0.85  # Thermal heat conductivity of the soil layers 	[W m^{-1} K^{-1}]

    
    x['tAir']      = ca.SX.sym('x_tAir'); 
    x['tTop']      = ca.SX.sym('x_tTop');

    u['heat']     = ca.SX.sym('u_heat');
    u['thScr']    = ca.SX.sym('u_thScr'); 

    d['tOut']     = ca.SX.sym('d_tOut');
    d['tSoOut']   = ca.SX.sym('d_tSoOut');
    d['wind']     = ca.SX.sym('d_wind');
    
    # Air flux through the thermal screen [m s^{-1}]
    a['fThScr'] = (1-u['thScr'])*0.82*0.12 + 2.4e-4
    
    # Between air in main and top compartment [W m^{-2}]
    a['hAirTop'] = p['rhoAir']*p['cPAir']*a['fThScr']*(x['tAir'] - x['tTop'])

    # Between top compartment and outside air [W m^{-2}]
    a['hTopOut'] = p['rhoAir']*p['cPAir']*p['cLeakage']*d['wind']*(x['tTop'] - d['tOut'])

    # Between air and soil [W m^{-2}]
    a['hAirSoOut'] = 2/(p['hSo1']/p['lambdaSo'])*(x['tAir'] - d['tSoOut']);

    dxdt['Air'] = 1/p['capAir']*(u['heat']-a['hAirSoOut']-a['hAirTop']);

    dxdt['Top'] = 1/p['capTop']*(a['hAirTop']-a['hTopOut']);

        
    # Stack states and dynamics
    x_vec = ca.vertcat(x['tAir'], x['tTop'])
    u_vec = ca.vertcat(u['thScr'], u['heat'])
    d_vec = ca.vertcat(d['tOut'], d['tSoOut'], d['wind'])
    dx_vec = ca.vertcat(dxdt['Air'], dxdt['Top'])
    
    # Continues time dynamics
    f = ca.Function('f', [x_vec, u_vec, d_vec], [dx_vec])
    
    return f, x_vec, u_vec, d_vec, dx_vec
