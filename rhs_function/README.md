## Governing equations

$$\frac{dY}{dt} = \frac{W_{k}}{\rho}\dot{\omega_{k}},\text{ where } k=1,2,...,N_s$$
$$\frac{dT}{dt} = -\frac{1}{\rho c_v} \sum_{k}W_{k}\dot{\omega_{k}}u_{k}$$

Reference: https://cantera.org/3.1/reference/reactors/ideal-gas-reactor.html

## Function description

state\_in: array of size $$N_s+1$$ with mass fractions and temperature.
Defining the thermodynamic state: TDY (constant density).
