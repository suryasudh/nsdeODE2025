from timestepping import timestepper, timestepper_ab
from rhs_code import rhs
from new_timestep import get_dt
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
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
    required_methods_ab=["ab1", "ab2"]


    t = np.array(0.0, dtype=dtype_time)
    tend=np.array(4.e-4,dtype=dtype_time)
    iter=0
    iter_interval=2
    dt=np.array(1.e-8, dtype=dtype_time)


    file_main_name = "../../output_64_rk1_rk2_adptv_run001_main.csv"
    file_ref_name = "../../output_64_rk1_rk2_adptv_run001_ref.csv"
    str_header = "H2,H,O2,OH,O,H2O,HO2,H2O2,N2,temp,time,dt"
    file_main = open(file_main_name, "w")
    file_ref = open(file_ref_name, "w")
    file_main.write(str_header + "\n")
    file_ref.write(str_header + "\n")

    class_of_methods = "rk" # "rk" or "ab"

    if class_of_methods == "ab":
        state_prev_1 = None
        state_prev_2 = None
        state_prev_3 = None
        state_prev_4 = None
        state_prev_5 = None


        while (t < tend) and iter < 5:
            if iter == 0:
                state_prev_1 = np.copy(state_arr)
            elif iter == 1:
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iter == 2:
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iter == 3:
                state_prev_4 = state_prev_3
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iter == 4:
                state_prev_5 = state_prev_4
                state_prev_4 = state_prev_3
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
                
            state_new = timestepper(tk=t, xk=state_arr, dt=dt, func=rhs, required_methods=required_methods,gas_obj=gas_obj,dens=dens_0,prec=dtype_global)
            state_arr_main = state_new[0, 0, :]
            state_arr_ref = state_new[1, 0, :]
            # print(state_new.shape)
            state_arr = state_arr_main

            t+=dt
            iter=iter+1

            # calling the function that computes dt
            # dt = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=1e-10,gamma=0.9,norm_type=2)
            dt = 5e-9

        while (t < tend):
            state_prev_5 = state_prev_4
            state_prev_4 = state_prev_3
            state_prev_3 = state_prev_2
            state_prev_2 = state_prev_1
            state_prev_1 = np.copy(state_arr_main)
  
            state_new = timestepper_ab(tk=0, xk=[state_prev_1, state_prev_2, state_prev_3, state_prev_4, state_prev_5], dt=dt, func=rhs, required_methods=required_methods_ab, gas_obj=gas_obj, dens=dens_0,prec=dtype_global)
            state_arr_main = state_new[0, 0, :]
            state_arr_ref = state_new[1, 0, :]

            t+=dt
            iter=iter+1

            # calling the function that computes dt
            # dt = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=1e-10,gamma=0.9,norm_type=2)
            dt = 5e-9



    elif class_of_methods == "rk":
        while (t < tend):      
            state_new = timestepper(tk=t, xk=state_arr, dt=dt, func=rhs, required_methods=required_methods,gas_obj=gas_obj,dens=dens_0,prec=dtype_global)
            state_arr_main = state_new[0, 0, :]
            state_arr_ref = state_new[1, 0, :]
            # print(state_new.shape)
            state_arr = state_arr_main

            t+=dt
            iter=iter+1

            # calling the function that computes dt
            dt = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=1e-10,gamma=0.9,norm_type=2)
            

            if iter%iter_interval==0:
                file_main.write(f"{state_arr_main[0]},{state_arr_main[1]},{state_arr_main[2]},{state_arr_main[3]},"+
                                f"{state_arr_main[4]},{state_arr_main[5]},{state_arr_main[6]},{state_arr_main[7]},"+
                                f"{state_arr_main[8]},{state_arr_main[9]},{t:.9e},{dt:.9e}\n")
            
                file_ref.write(f"{state_arr_ref[0]},{state_arr_ref[1]},{state_arr_ref[2]},{state_arr_ref[3]},"+
                               f"{state_arr_ref[4]},{state_arr_ref[5]},{state_arr_ref[6]},{state_arr_ref[7]},"+
                               f"{state_arr_ref[8]},{state_arr_ref[9]},{t:.9e},{dt:.9e}\n")



