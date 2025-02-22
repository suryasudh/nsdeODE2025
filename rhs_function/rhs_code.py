import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

def rhs(t,state_in,gas):
    Neq = len(state_in)
    Y = state_in[0:Neq-1]
    T = state_in[Neq-1]
    gas.TDY = T,rho0,Y[:]
    rhs_out = np.copy(state_in)
    rhs_out[0:Neq-1] = gas.net_production_rates*gas.molecular_weights/gas.density
    for i in range(gas.n_species):
        rhs_out[Neq-1] = rhs_out[Neq-1] - gas.net_production_rates[i]*gas.standard_int_energies_RT[i]*ct.gas_constant*gas.T
    rhs_out[Neq-1] = rhs_out[Neq-1]/(gas.cp_mass*gas.density)
    return rhs_out
