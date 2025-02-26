from timestepping import timestepper
from rhs_code import rhs
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Define the initial conditions
    gas_obj = ct.Solution("../yaml_files/h2_sandiego.yaml")
    T0 = 1000.
    P0 = 101325.
    X0 = "H2:2.,O2:1.,N2:3.76"
    gas_obj.TPX = T0,P0,X0
    state_arr = np.hstack((gas_obj.Y,gas_obj.T)).ravel()
    dens_0 = gas_obj.density_mass

    t=0.
    tend=5.e-4
    iter=0
    iter_interval=100
    dt=1.e-8
    # niters = int(tend/dt)
    T_arr = []
    T_arr_RK1 = []
    YOH_arr = []
    while t<tend:
        state_new = timestepper(tk=t, xk=state_arr, dt=dt, func=rhs, required_methods=["rk4", "rk1"],gas_obj=gas_obj,dens=dens_0)
        state_arr = state_new[0, 0, :]
        state_arr2 = state_new[1, 0, :]
        # print(state_new.shape)

        t+=dt
        iter=iter+1
        if iter%iter_interval==0:
            # print(f"Time={t:.4e}, Temperature={state_arr[-1]:.4f}")
            T_arr.append(state_arr[-1])
            T_arr_RK1.append(state_arr2[-1])
            YOH_arr.append(state_arr[4])
        # break
        # calling the function that computes dt


    plt.plot(T_arr,label="RK4")
    plt.plot(T_arr_RK1,label="RK1")
    plt.legend()
    plt.show()

    # plt.plot(YOH_arr)
    # plt.show()

