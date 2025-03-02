from timestepping import timestepper
from rhs_code import rhs
from new_timestep import get_dt
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
dtype_global = np.float64
dtype_time = np.float64

if __name__ == '__main__':
    # Define the initial conditions
    gas_obj = ct.Solution("../yaml_files/h2_sandiego.yaml")
    T0 = 1000.
    P0 = 101325.
    X0 = "H2:2.,O2:1.,N2:3.76"
    gas_obj.TPX = T0,P0,X0
    state_arr = np.hstack((gas_obj.Y,gas_obj.T)).ravel().astype(dtype=dtype_global)
    dens_0 = gas_obj.density_mass
    required_methods=["rk1", "rk2"]


    t = np.array(0.0, dtype=dtype_time)
    tend=np.array(4.e-4,dtype=dtype_time)
    iter=0
    iter_interval=2
    dt=np.array(1.e-8, dtype=dtype_time)
    # niters = int(tend/dt)
    T_arr_RK1 = []
    T_arr_RK2 = []
    YOH_arr = []
    YH_arr = []
    dts_arr = []
    ts_arr = []
    while t<tend:
        state_new = timestepper(tk=t, xk=state_arr, dt=dt, func=rhs, required_methods=required_methods,gas_obj=gas_obj,dens=dens_0)
        state_arr_main = state_new[0, 0, :]
        state_arr_ref = state_new[1, 0, :]
        # print(state_new.shape)
        state_arr = state_arr_main

        t+=dt
        iter=iter+1
        if iter%iter_interval==0:
            # print(f"Time={t:.4e}, Temperature={state_arr[-1]:.4f}")
            ts_arr.append(np.copy(t))
            # print(f"Time={t:.4e}")
            T_arr_RK1.append(state_arr[-1])
            T_arr_RK2.append(state_arr_ref[-1])
            YOH_arr.append(state_arr[4])
            YH_arr.append(state_arr[1])
        # break
        # calling the function that computes dt
        dt = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=1e-10,gamma=0.9,norm_type=2)
        # print("dt: ",dt)
        dts_arr.append(dt)
    
    plt.plot(ts_arr,T_arr_RK1,label="RK1")
    plt.plot(ts_arr,T_arr_RK2,label="RK2")
    plt.legend()
    plt.show()

    plt.plot(ts_arr,YOH_arr)
    plt.show()

    plt.plot(ts_arr,YH_arr)
    plt.show()

    plt.plot(dts_arr)
    plt.show()

