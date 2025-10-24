import numpy as np
import casadi as ca

params = [
    5.4400e-01,
    2.6500e-07,
    5.3000e01,
    3.5500e-09,
    5.1100e-06,
    2.3000e-04,
    6.2900e-04,
    5.2000e-05,
    4.1000e00,
    4.8700e-07,
    7.5000e-06,
    8.3100e00,
    2.7315e02,
    1.0132e05,
    4.4000e-02,
    3.0000e04,
    1.2900e03,
    6.1000e00,
    2.0000e-01,
    4.1000e00,
    3.6000e-03,
    9.3480e03,
    8.3140e03,
    2.7315e02,
    1.7400e01,
    2.3900e02,
    1.7269e01,
    2.3830e02,
]

def odes_screen_lin(p):
    # Linear interaction matrix
    M = np.zeros((7, 7))
    M[1, 1] = -0.5 * (1 / p[8]) 
    M[1, 4] = 0.5 * (1 / p[8]) 
    M[4, 4] = -0.5 * (1 / p[8]) 
    M[4, 1] = 0.5 * (1 / p[8]) 

    # M[2, 2] = -0.5 * (1 / p[8]) 
    # M[2, 5] = 0.5 * (1 / p[8]) 
    # M[5, 5] = -0.5 * (1 / p[8]) 
    # M[5, 2] = 0.5 * (1 / p[8]) 
    M[2, 2] = -0.5 * (p[16] / p[15]) 
    M[2, 5] = 0.5 * (p[16] / p[15]) 
    M[5, 5] = -0.5 * (p[16] / p[15]) 
    M[5, 2] = 0.5 * (p[16] / p[15]) 

    M[3, 3] = -0.5 * (1 / p[19]) 
    M[3, 6] = 0.5 * (1 / p[19]) 
    M[6, 6] = -0.5 * (1 / p[19]) 
    M[6, 3] = 0.5 * (1 / p[19]) 

    return M

def odes_screen_nonlin(x, u, d, p):
    """
    Computes the derivatives of the Van Henten greenhouse climate model.

    Parameters (not right!):
    - x: State vector [drymass, Air_CO2, Air_temp, Air_humidity]
    - u: Control inputs [CO2_supply, Ventilation_rate, Heating_energy]
    - d: Disturbances [Radiation, Outside_CO2, Outside_temp, Outside_humidity]
    - p: Parameter vector (length 28)

    Returns:
    - dxdt: Derivatives of the state variables
    """

    # Gross canopy photosynthesis rate denominator
    phi = p[3] * d[0] + (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7])

    # Gross canopy photosynthesis rate (PhiPhot_c)
    PhiPhot_c = (
        (1 - np.exp(-p[2] * x[0]))
        * (p[3] * d[0] * (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7]))
        / phi
    )

    # Mass exchange of CO2 through the vents (PhiVent_c)
    PhiVent_c = (u[1] * 1e-3 + p[10]) * (x[4] - d[1])

    # Mass exchange of H2O through the vents (PhiVent_h)
    PhiVent_h = (u[1] * 1e-3 + p[10]) * (x[6] - d[3])

    # Canopy transpiration (PhiTransp_h)
    PhiTransp_h = (
        p[20]
        * (1 - np.exp(-p[2] * x[0]))
        * (
            (p[21] / (p[22] * (x[2] + p[23]))) * np.exp(p[24] * x[2] / (x[2] + p[25]))
            - x[3]
        )
    )

    # Differential equations for nonlinear components
    dx1dt = p[0] * PhiPhot_c - p[1] * x[0] * 2 ** (x[2] / 10 - 2.5)
    dx2dt = (
        0.5
        * (1 / p[8])
        * (-PhiPhot_c + p[9] * x[0] * 2 ** (x[2] / 10 - 2.5) + u[0] * 1e-6)
    )
    dx3dt = 0.5 * (1 / p[15]) * (u[2] + p[18] * d[0])
    dx4dt = 0.5 * (1 / p[19]) * (PhiTransp_h)

    dx5dt = 0.5 * (1 / p[8]) * (-PhiVent_c)
    dx6dt = 0.5 * (1 / p[15]) * (-(p[16] * u[1] * 1e-3 + p[17]) * (x[5] - d[2]))
    dx7dt = 0.5 * (1 / p[19]) * (-PhiVent_h)

    return ca.vertcat(*[dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt])

def h_meas_screen(x, p):
    """
    Measurement function for greenhouse climate model.

    Parameters:
    - x: State vector [Dry weight, Air_CO2, Air_temp, Air_humidity]
    - p: Parameter vector (length 28)

    Returns:
    - y: Output vector [Weight (g/m^2), CO2 (ppm), Air Temp (°C), RH (%)]
    """
    y_1 = 1e3 * x[0]  # Weight in g/m^2
    y_2 = p[11] * (x[2] + p[12]) / (p[13] * p[14]) * x[1] * 1e3  # CO2 in ppm
    y_3 = x[2]  # Air temperature in °C
    y_4 = (
        p[11]
        * (x[2] + p[12])
        / (11 * np.exp(p[26] * x[2] / (x[2] + p[27])))
        * x[3]
        * 1e2
    )  # RH in %
    y_5 = p[11] * (x[2] + p[12]) / (p[13] * p[14]) * x[4] * 1e3  # CO2 in ppm
    y_6 = x[5]
    y_7 = (
        p[11]
        * (x[5] + p[12])
        / (11 * np.exp(p[26] * x[5] / (x[5] + p[27])))
        * x[6]
        * 1e2
    )  # RH in %

    return ca.vertcat(*[y_1, y_2, y_3, y_4, y_5, y_6, y_7])