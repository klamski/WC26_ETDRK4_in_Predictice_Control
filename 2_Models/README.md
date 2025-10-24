## `RK4_integration.py`
Runge-Kutta 4 integration method receiving as inputs the time interval h, the symbolic state vector x, the symbolic input vector u, the symbolic disturbance vector d and the symbolic function f (in this case the lettuce model). 

## `collocation.py`
Python code implementing the collocation integration method receiving as inputs the time interval h, the dimension of the state vector (nx), the dimension of the input vector (nu), the dimension of the disturbance vector (nd) and the symbolic function f (in this case the lettuce model). 

## `lettuce_model.py`
The climate-crop lettuce model as originally presented in [1] written in a symbolic form using casAdi [2].

## `stiff_dynamics.py`
Simplified model simulating the some of the thermal dynamics of a greenhouse using **first-principles energy balance** ODEs built with [CasADi](https://web.casadi.org/). It describes the temperature evolution in a two-layer greenhouse: the main air compartment and the top air compartment. The two compartments can be separated by deploying a thermal screen. 

### System Description

- **Main Air Compartment Temperature (`tAir`)**
- **Top Air Compartment Temperature (`tTop`)**

#### Model Equations

The state evolution is governed by:

$$
\begin{aligned}
\frac{d\mathrm{T}_{\mathrm{air}}}{dt} &= \frac{1}{\mathrm{C}_{\mathrm{air}}} \left( Q_{\mathrm{heat}} - H_{\mathrm{AirSoil}} - H_{\mathrm{AirTop}} \right) \\\\
\frac{d\mathrm{T}_{\mathrm{top}}}{dt} &= \frac{1}{\mathrm{C}_{\mathrm{top}}} \left( H_{\mathrm{AirTop}} - H_{\mathrm{TopOut}} \right)
\end{aligned}
$$

Where:

- $Q_{\mathrm{heat}}$ :  heating input  
- $H_{\mathrm{AirSoil}}$ : heat transfer from air to soil  
- $H_{\mathrm{AirTop}}$ : convective exchange between air and top layer  
- $H_{\mathrm{TopOut}}$ : heat loss to the outside air


#### Parameters

| Parameter     | Description                                    | Value       |
|---------------|------------------------------------------------|-------------|
| `rhoAir`      | Density of air                                 | 1.2 kg/m³   |
| `cPAir`       | Specific heat capacity of air                  | 1000 J/kg·K |
| `capAir`      | Heat capacity of main air compartment          | 7560 J/K·m² |
| `capTop`      | Heat capacity of top compartment               | 726 J/K·m²  |
| `cLeakage`    | Greenhouse leakage coefficient                 | 1e-4        |
| `hSo1`        | Soil layer thickness                           | 0.04 m      |
| `lambdaSo`    | Soil thermal conductivity                      | 0.85 W/m·K  |

#### Model Inputs and States

#### Control Inputs (`u`)
- `u_thScr`: thermal screen opening fraction (0–1)
- `u_heat`: heating power input [W/m²]

#### Disturbances (`d`)
- `tOut`: outside air temperature [°C]
- `tSoOut`: soil surface temperature [°C]
- `wind`: wind speed [m/s]

#### States (`x`)
- `tAir`: air temperature inside the greenhouse [°C]
- `tTop`: top layer air temperature [°C]


The system dynamics are implemented symbolically via CasADi:




## References
[1] van Henten, E.J. (1994). Greenhouse climate management: an optimal control approach. Ph.D. thesis, Wageningen
University & Research, Wageningen, Netherlands. PhD
Dissertation

[2] Andersson, J., Gillis, J., Horn, G., Rawlings, J., and Diehl,
M. (2018). *Casadi: a software framework for nonlinear optimization and optimal control.* Mathematical Pro-
gramming Computation, 11.
