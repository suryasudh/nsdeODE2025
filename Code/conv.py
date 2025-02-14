import sys
import numpy as np

import cantera as ct

# The simulation integrates in time for these values:
nstep = 15000
step_size = 5.e-8

# Defines the homogeneous mixture properties for use in the reactor
# COH2.cti is the chemical input file generated from the CHEMKIN files

#To generate the .cti file do: (see> man ck2cti)
#	ck2cti --input=chem.inp --thermo=therm.dat --transport=trans.dat 

gas = ct.Solution('H2_sandiego.yaml')
# Sets the initial temperature, pressure and composition of the mixture
# .TPX used mole fractions, .TPY uses mass fractions
# absent species are set to 0.0
# All values is SI units
gas.set_equivalence_ratio(phi=1., fuel="H2:1.", oxidizer="O2:0.21, N2:0.79")


gas.TP =1000.0, ct.one_atm

# 'r' is the name of the object which repreesntes your reactor
# Here, r is a constant pressure reactor which solves the conservation 
# of species and energy equation
# The mixture properties are passed into the call to set the initial condition
r = ct.IdealGasReactor(gas)

# sim is the object which represents the progress in time of the reactor
sim = ct.ReactorNet([r])
time = 0.0


#Here output is just to screen, can call routines to save to .csv or other files
# see numpy documentation

# To get ignition delay time, store array of T and time, define ignition as, say
# max(dT/dt)

print(gas.Y[:])

dtdT = np.zeros(shape=(nstep,7))

print("Printing cantera solver initial conditions:")

print("mass fractions:",gas.Y)
print("Temperature:",gas.T)
print("Pressure:",gas.P)
print("Density:",gas.density)
print("Velocity:",0.)
print("gamma:",gas.cp_mole/gas.cv_mole)
print("heat capacity cp:",gas.cp_mass)
print("heat capacity cv:",gas.cv_mass)
print("viscosity mu:",gas.viscosity)
print("production rates:",gas.net_production_rates)
print("sensible internal energy:",gas.int_energy_mass)
print("thermal conductivity:",gas.thermal_conductivity)

print("Printing cantera solver initial conditions done")

print('%10s %10s %10s %14s' % ('t [s]','T [K]','P [Pa]','u [J/kg]'))
for n in range(nstep):
    time += step_size
# Integrates the solution in time
    sim.advance(time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T,
                                           r.thermo.P, r.thermo.u))

    dtdT[n,0] = time
    dtdT[n,1] = r.T
    dtdT[n,2] = r.Y[0]
    dtdT[n,3] = r.Y[6]
    dtdT[n,4] = r.Y[5]
    dtdT[n,5] = r.Y[4]
    dtdT[n,6] = r.Y[1]


# write dtdT[:,0] to file named "time.txt"
# write dtdT[:,1] to file named "temperature.txt"
# write dtdT[:,2] to file named "H2.txt"
# write dtdT[:,3] to file named "HO2.txt"
# write dtdT[:,4] to file named "H2O.txt"

np.savetxt("time.txt",dtdT[:,0])
np.savetxt("temperature.txt",dtdT[:,1])
np.savetxt("H2.txt",dtdT[:,2])
np.savetxt("HO2.txt",dtdT[:,3])
np.savetxt("H2O.txt",dtdT[:,4])

# max_index = np.argmax(np.gradient(dtdT[:,1],np.gradient(dtdT[:,0])))

# print(dtdT[max_index,0])

print("Printing cantera solver final state:")

print("mass fractions:",gas.Y)
print("Temperature:",gas.T)
print("Pressure:",gas.P)
print("Density:",gas.density)
print("Velocity:",0.)
print("gamma:",gas.cp_mole/gas.cv_mole)
print("heat capacity cp:",gas.cp_mass)
print("heat capacity cv:",gas.cv_mass)
print("viscosity mu:",gas.viscosity)
print("production rates:",gas.net_production_rates)
print("sensible internal energy:",gas.int_energy_mass)
print("thermal conductivity:",gas.thermal_conductivity)

print("Printing cantera solver final state done")


# Plot the results if matplotlib is installed.
# See http://matplotlib.org/ to get it.
if '--plot' in sys.argv[1:]:
    import matplotlib.pyplot as plt
    plt.plot(dtdT[:,0],dtdT[:,1])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Temperature (K)',fontsize=14)
    plt.show()

    plt.plot(dtdT[:,0],dtdT[:,2])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel(r'$Y_{H_2}$',fontsize=14)
    plt.show()

    plt.plot(dtdT[:,0],dtdT[:,3])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel(r'$Y_{HO_2}$',fontsize=14)
    plt.show()

    plt.plot(dtdT[:,0],dtdT[:,4])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel(r'$Y_{H_2O}$',fontsize=14)
    plt.show()

    plt.plot(dtdT[:,0],dtdT[:,5])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel(r'$Y_{O}$',fontsize=14)
    plt.show()

    plt.plot(dtdT[:,0],dtdT[:,6])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel(r'$Y_{H}$',fontsize=14)
    plt.show()
# Template for plotting values of interest
#    plt.clf()
#    plt.plot(,)
#    plt.xlabel('')
#    plt.ylabel('')
#    plt.tight_layout()
#    plt.show()
else:
    print("To view a plot of these results, run this script with the option --plot")
