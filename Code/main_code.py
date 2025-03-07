from timestepping import timestepper, timestepper_ab
from rhs_code import rhs
from new_timestep import get_dt
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from time_and_memory import measure_time_and_memory
from tqdm import tqdm

def run_simulation(class_of_methods, main_num, ref_num, precision_str:str, 
                   time_step:str, time_step_val=None, tolerance:float=1e-10, run_num:str="001", initial_temp:float=1000.0, io_flag:bool=False):
    dtype_global = None
    dtype_time = None


    if precision_str == "64":
        dtype_global = np.float64
        dtype_time = np.float64
    elif precision_str == "32":
        dtype_global = np.float32
        dtype_time = np.float32
    elif precision_str == "16":
        dtype_global = np.float16
        dtype_time = np.float16
    else:
        raise ValueError("Invalid precision string")
    gas_obj = ct.Solution("../yaml_files/h2_sandiego.yaml")
    T0 = initial_temp                              # Initial temperature in K
    P0 = 101325.
    X0 = "H2:2.,O2:1.,N2:3.76"
    gas_obj.TPX = T0,P0,X0
    state_arr = np.hstack((gas_obj.Y,gas_obj.T)).ravel().astype(dtype=dtype_global)
    dens_0 = gas_obj.density_mass


    if time_step == "const" and time_step_val is None:
        raise AssertionError("Time step value needs to be specified for constant time stepping")
    

    required_methods=[f"rk{main_num}", f"rk{ref_num}"]
    required_methods_ab=[f"ab{main_num}", f"ab{ref_num}"]   # both are needed because ab is not self starting

    if io_flag:
        file_main_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_{time_step}_run{run_num}_main.csv"
        file_ref_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_{time_step}_run{run_num}_ref.csv"
        str_header = f"H2,H,O2,OH,O,H2O,HO2,H2O2,N2,temp,time,dt,error,iter"
        file_main = open(file_main_name, "w")
        file_ref = open(file_ref_name, "w")
        file_main.write(str_header + "\n")
        file_ref.write(str_header + "\n")


    t = np.array(0.0, dtype=dtype_time)
    tend=np.array(4.e-4,dtype=dtype_time)
    iter=0
    iter_interval=2
    dt=np.array(1.e-8, dtype=dtype_time)


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

            #############################################################
            # Ensuring that the species mass fractions are not negative
            ##############################################################
            state_arr_main[:9] = np.maximum(state_arr_main[:9], 1.0e-16)
            state_arr_ref[:9] = np.maximum(state_arr_ref[:9], 1.0e-16)
            # print(state_new.shape)
            state_arr = state_arr_main

            t+=dt
            iter=iter+1

            # initiating with a fixed dt
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

            #############################################################
            # Ensuring that the species mass fractions are not negative
            ##############################################################
            state_arr_main[:9] = np.maximum(state_arr_main[:9], 1.0e-16)
            state_arr_ref[:9] = np.maximum(state_arr_ref[:9], 1.0e-16)

            t+=dt
            iter=iter+1

            # calling the function that computes dt
            if time_step == "const":
                _, error = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=tolerance,gamma=0.9,norm_type=2)
                dt = time_step_val
            elif time_step == "adptv":
                dt, error = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=tolerance,gamma=0.9,norm_type=2)

            if io_flag:
                if iter%iter_interval==0:
                    file_main.write(f"{state_arr_main[0]},{state_arr_main[1]},{state_arr_main[2]},{state_arr_main[3]},"+
                                    f"{state_arr_main[4]},{state_arr_main[5]},{state_arr_main[6]},{state_arr_main[7]},"+
                                    f"{state_arr_main[8]},{state_arr_main[9]},{t:.9e},{dt:.9e},{error:.9e},{iter}\n")

                    file_ref.write(f"{state_arr_ref[0]},{state_arr_ref[1]},{state_arr_ref[2]},{state_arr_ref[3]},"+
                                   f"{state_arr_ref[4]},{state_arr_ref[5]},{state_arr_ref[6]},{state_arr_ref[7]},"+
                                   f"{state_arr_ref[8]},{state_arr_ref[9]},{t:.9e},{dt:.9e},{error:.9e},{iter}\n")



    elif class_of_methods == "rk":
        while (t < tend):      
            state_new = timestepper(tk=t, xk=state_arr, dt=dt, func=rhs, required_methods=required_methods,gas_obj=gas_obj,dens=dens_0,prec=dtype_global)
            state_arr_main = state_new[0, 0, :]
            state_arr_ref = state_new[1, 0, :]

            #############################################################
            # Ensuring that the species mass fractions are not negative
            ##############################################################
            state_arr_main[:9] = np.maximum(state_arr_main[:9], 1.0e-16)
            state_arr_ref[:9] = np.maximum(state_arr_ref[:9], 1.0e-16)
            # print(state_new.shape)
            state_arr = state_arr_main

            t+=dt
            iter=iter+1

            if time_step == "const":
                _, error = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=tolerance,gamma=0.9,norm_type=2)
                dt = time_step_val
            elif time_step == "adptv":
                dt, error = get_dt(dt_old=dt, main_method=required_methods[0], x_main=state_arr_main, x_ref=state_arr_ref, tolerance=tolerance,gamma=0.9,norm_type=2)

            if io_flag:
                if iter%iter_interval==0:
                    file_main.write(f"{state_arr_main[0]},{state_arr_main[1]},{state_arr_main[2]},{state_arr_main[3]},"+
                                    f"{state_arr_main[4]},{state_arr_main[5]},{state_arr_main[6]},{state_arr_main[7]},"+
                                    f"{state_arr_main[8]},{state_arr_main[9]},{t:.9e},{dt:.9e},{error:.9e},{iter}\n")

                    file_ref.write(f"{state_arr_ref[0]},{state_arr_ref[1]},{state_arr_ref[2]},{state_arr_ref[3]},"+
                                   f"{state_arr_ref[4]},{state_arr_ref[5]},{state_arr_ref[6]},{state_arr_ref[7]},"+
                                   f"{state_arr_ref[8]},{state_arr_ref[9]},{t:.9e},{dt:.9e},{error:.9e},{iter}\n")
    if io_flag:
        file_main.close()
        file_ref.close()





if __name__ == '__main__':
    for initial_temp in tqdm([1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0]):
        # Define the initial conditions
        class_of_methods = "ab"
        main_num = 4
        ref_num = 5
        precision_str = "32"
        time_step = "adptv"
        tolerance = 1.e-9
        # initial_temp = 1000.0

        run_num = None
        if initial_temp == 1000.0:
            run_num = "001"
        elif initial_temp == 1050.0:
            run_num = "002"
        elif initial_temp == 1100.0:
            run_num = "003"
        elif initial_temp == 1150.0:
            run_num = "004"
        elif initial_temp == 1200.0:
            run_num = "005"
        elif initial_temp == 1250.0:
            run_num = "006"
        elif initial_temp == 1300.0:
            run_num = "007"
        elif initial_temp == 1350.0:
            run_num = "008"
        

        logging_file_name = f"../../logging_resultsab32.txt"
        logging_file = open(logging_file_name, "a")
        logging_file.write("======== RUN SUMMARY ========\n")
        logging_file.write(f"Initial temperature: {initial_temp}\n")
        logging_file.write(f"Precision: {precision_str}\n")
        logging_file.write(f"Main: {class_of_methods}{main_num}, Ref:{class_of_methods}{ref_num}, dt_initial={time_step}, run={run_num}\n")

        ###########################################################
        ### ADAPTIVE TIME STEPPING
        ###########################################################
        run_simulation(class_of_methods=class_of_methods, main_num=main_num, ref_num=ref_num, 
                    precision_str=precision_str, time_step=time_step, time_step_val=None, tolerance=tolerance, run_num=run_num, initial_temp=initial_temp, io_flag=True)
        
        (_, adapt_time, adapt_cur_mem, adapt_peak_mem) = measure_time_and_memory(
            run_simulation, class_of_methods=class_of_methods, main_num=main_num, ref_num=ref_num, 
                    precision_str=precision_str, time_step=time_step, time_step_val=None, tolerance=tolerance, run_num=run_num, initial_temp=initial_temp, io_flag=False
        )

        logging_file.write(f"*** ADAPTIVE DT RUN ***\n")
        logging_file.write(f"perf_run_time: {adapt_time:.6f} s, Current Memory: {adapt_cur_mem} bytes, Peak Memory: {adapt_peak_mem} bytes\n")

        file_main_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_{time_step}_run{run_num}_main.csv"
        df_main = pd.read_csv(file_main_name, dtype="float64")

        dt_iter_max = df_main["iter"].max()        
        logging_file.write(f"Maximum iterations: {dt_iter_max}\n")

        dt_min = df_main["dt"].min()
        logging_file.write(f"Minimum dt: {dt_min} from adaptive method (used for const timestepping below)\n")

        ###########################################################
        ### CONSTANT TIME STEPPING
        ###########################################################
        logging_file.write(f"*** CONSTANT DT RUN ***\n")

        time_step_val = dt_min
        time_step = "const"

        run_simulation(class_of_methods=class_of_methods, main_num=main_num, ref_num=ref_num, 
                    precision_str=precision_str, time_step=time_step, time_step_val=time_step_val, tolerance=tolerance, initial_temp=initial_temp, run_num=run_num,io_flag=True)

        (_, const_time, const_cur_mem, const_peak_mem) = measure_time_and_memory(
            run_simulation, class_of_methods=class_of_methods, main_num=main_num, ref_num=ref_num, 
                    precision_str=precision_str, time_step=time_step, time_step_val=time_step_val, tolerance=tolerance, initial_temp=initial_temp, run_num=run_num,io_flag=False
        )

        logging_file.write(f"perf_run_time: {const_time:.6f} s, Current Memory: {const_cur_mem} bytes, Peak Memory: {const_peak_mem} bytes\n")

        file_main_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_{time_step}_run{run_num}_main.csv"
        df_main = pd.read_csv(file_main_name, dtype="float64")

        dt_iter_max = df_main["iter"].max()        
        logging_file.write(f"Maximum iterations: {dt_iter_max}\n")

        logging_file.write("======== END OF RUN ========\n\n")
        logging_file.close()