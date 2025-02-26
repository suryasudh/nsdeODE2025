import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

# TODO: include argument for specifying precision
def rhs(t,state_in,gas,rho0): #variable rho0 assumed to be a global float
    Neq = len(state_in)
    # print("Neq = ",Neq)
    Y = state_in[0:Neq-1]
    T = state_in[Neq-1]
    gas.TDY = T,rho0,Y[:]
    rhs_out = np.copy(state_in)
    rhs_out[0:Neq-1] = gas.net_production_rates*gas.molecular_weights/gas.density
    rhs_out[Neq-1] = -1.*ct.gas_constant*gas.T/(gas.cv_mass*gas.density)*np.dot(gas.net_production_rates,gas.standard_int_energies_RT)
    return rhs_out
