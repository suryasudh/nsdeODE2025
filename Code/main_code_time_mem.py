from timestepping import timestepper, timestepper_ab
from rhs_code import rhs
from new_timestep import get_dt
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os

from time_and_memory import measure_time_and_memory

dtype_global = None
dtype_time = None

##############################################################
### SELECTIONS
##############################################################
precision_str = "64"

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


def run_simulation_with_adaptive_dt():
    """
    Runs the simulation with ADAPTIVE dt for Adams-Bashforth.
    Returns the list of dt values used in all steps.
    """
    gas_obj = ct.Solution("../yaml_files/h2_sandiego.yaml")
    T0 = 1000.0
    P0 = 101325.0
    X0 = "H2:2.,O2:1.,N2:3.76"
    gas_obj.TPX = T0, P0, X0
    state_arr = np.hstack((gas_obj.Y, gas_obj.T)).astype(dtype_global)
    dens_0 = gas_obj.density_mass

    class_of_methods = "ab"   # or "rk"
    main_num, ref_num = "1", "2"

    # For the Adams-Bashforth approach, we still need an RK method to start it up
    # (the user can pick e.g. "rk1" (Euler) or "rk4", etc.).
    required_methods = [f"rk{main_num}", f"rk{ref_num}"]     # for startup
    required_methods_ab = [f"ab{main_num}", f"ab{ref_num}"]  # for the main AB steps

    file_main_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_adptv_run001_main.csv"
    file_ref_name = f"../../output_{precision_str}_{class_of_methods}{main_num}_{class_of_methods}{ref_num}_adptv_run001_ref.csv"
    str_header = "H2,H,O2,OH,O,H2O,HO2,H2O2,N2,temp,time,dt"

    file_main = open(file_main_name, "w")
    file_ref = open(file_ref_name, "w")
    file_main.write(str_header + "\n")
    file_ref.write(str_header + "\n")

    # Time loop setup
    t = np.array(0.0, dtype=dtype_time)
    tend = np.array(4.e-4, dtype=dtype_time)
    iteration = 0
    iter_interval = 2

    # Initial guess for dt
    dt = np.array(1.e-8, dtype=dtype_time)

    # Collect dt's in a list
    dt_values = []

    # ---------------------------------------------------------------------
    #  Adams-Bashforth method requires multi-step startup
    # ---------------------------------------------------------------------
    if class_of_methods == "ab":
        # We'll do the first 5 steps with a standard RK approach:
        state_prev_1 = None
        state_prev_2 = None
        state_prev_3 = None
        state_prev_4 = None
        state_prev_5 = None

        # Startup
        while (t < tend) and (iteration < 5):
            if iteration == 0:
                state_prev_1 = np.copy(state_arr)
            elif iteration == 1:
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iteration == 2:
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iteration == 3:
                state_prev_4 = state_prev_3
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)
            elif iteration == 4:
                state_prev_5 = state_prev_4
                state_prev_4 = state_prev_3
                state_prev_3 = state_prev_2
                state_prev_2 = state_prev_1
                state_prev_1 = np.copy(state_arr)

            state_new = timestepper(
                tk=t,
                xk=state_arr,
                dt=dt,
                func=rhs,
                required_methods=required_methods,  # e.g. [rk1, rk2]
                gas_obj=gas_obj,
                dens=dens_0,
                prec=dtype_global
            )

            state_arr_main = state_new[0, 0, :]
            state_arr_ref  = state_new[1, 0, :]

            # Update the "main" state
            state_arr = state_arr_main
            t += dt
            iteration += 1

            # Use get_dt for next step (instead of the old fixed 5e-9)
            dt = get_dt(
                dt_old=dt,
                main_method=required_methods[0],  # e.g. "rk1"
                x_main=state_arr_main,
                x_ref=state_arr_ref,
                tolerance=1e-10,
                gamma=0.9,
                norm_type=2
            )
            dt_values.append(dt)

            # Write to file if desired
            if iteration % iter_interval == 0:
                file_main.write(
                    f"{state_arr_main[0]},{state_arr_main[1]},{state_arr_main[2]},{state_arr_main[3]},"
                    f"{state_arr_main[4]},{state_arr_main[5]},{state_arr_main[6]},{state_arr_main[7]},"
                    f"{state_arr_main[8]},{state_arr_main[9]},{t:.9e},{dt:.9e}\n"
                )
                file_ref.write(
                    f"{state_arr_ref[0]},{state_arr_ref[1]},{state_arr_ref[2]},{state_arr_ref[3]},"
                    f"{state_arr_ref[4]},{state_arr_ref[5]},{state_arr_ref[6]},{state_arr_ref[7]},"
                    f"{state_arr_ref[8]},{state_arr_ref[9]},{t:.9e},{dt:.9e}\n"
                )

        # Now switch to the AB method
        while t < tend:
            # Shift the old states forward
            state_prev_5 = state_prev_4
            state_prev_4 = state_prev_3
            state_prev_3 = state_prev_2
            state_prev_2 = state_prev_1
            state_prev_1 = np.copy(state_arr_main)

            # Evaluate next step with AB
            state_new = timestepper_ab(
                tk=t,  # or 0 if you prefer
                xk=[state_prev_1, state_prev_2, state_prev_3, state_prev_4, state_prev_5],
                dt=dt,
                func=rhs,
                required_methods=required_methods_ab,  # e.g. [ab1, ab2]
                gas_obj=gas_obj,
                dens=dens_0,
                prec=dtype_global
            )
            state_arr_main = state_new[0, 0, :]
            state_arr_ref  = state_new[1, 0, :]

            t += dt
            iteration += 1

            # Also do adaptive dt inside the AB steps
            dt = get_dt(
                dt_old=dt,
                main_method=required_methods[0],  # e.g. "rk1"
                x_main=state_arr_main,
                x_ref=state_arr_ref,
                tolerance=1e-10,
                gamma=0.9,
                norm_type=2
            )
            dt_values.append(dt)

            # Write output every few steps
            if iteration % iter_interval == 0:
                file_main.write(
                    f"{state_arr_main[0]},{state_arr_main[1]},{state_arr_main[2]},{state_arr_main[3]},"
                    f"{state_arr_main[4]},{state_arr_main[5]},{state_arr_main[6]},{state_arr_main[7]},"
                    f"{state_arr_main[8]},{state_arr_main[9]},{t:.9e},{dt:.9e}\n"
                )
                file_ref.write(
                    f"{state_arr_ref[0]},{state_arr_ref[1]},{state_arr_ref[2]},{state_arr_ref[3]},"
                    f"{state_arr_ref[4]},{state_arr_ref[5]},{state_arr_ref[6]},{state_arr_ref[7]},"
                    f"{state_arr_ref[8]},{state_arr_ref[9]},{t:.9e},{dt:.9e}\n"
                )

    # ---------------------------------------------------------------------
    # If you used "rk" directly, you'd do a single while loop with get_dt each step.
    # ---------------------------------------------------------------------

    file_main.close()
    file_ref.close()
    return dt_values


def run_with_dt_minimum(dt_minimum):
    """
    (Optional) Show how to run the same system with a constant dt_minimum.
    """
    gas_obj = ct.Solution("../yaml_files/h2_sandiego.yaml")
    T0 = 1000.0
    P0 = 101325.0
    X0 = "H2:2.,O2:1.,N2:3.76"
    gas_obj.TPX = T0, P0, X0
    dens_0 = gas_obj.density_mass

    required_methods = ["rk4", "rk5"]  # just for demonstration
    state_arr = np.hstack((gas_obj.Y, gas_obj.T)).astype(dtype_global)

    tstart = 0.0
    tend = 4.e-4
    t = tstart

    while t < tend:
        state_new = timestepper(
            tk=t,
            xk=state_arr,
            dt=dt_minimum,
            func=rhs,
            required_methods=required_methods,
            gas_obj=gas_obj,
            dens=dens_0,
            prec=dtype_global
        )
        state_arr = state_new[0, 0, :]
        t += dt_minimum

    return state_arr


if __name__ == "__main__":
    # Measure time/memory for the ADAPTIVE run
    (dt_values, adapt_time, adapt_cur_mem, adapt_peak_mem) = measure_time_and_memory(
        run_simulation_with_adaptive_dt
    )
    print("======== ADAPTIVE RUN ========")
    print(f"Total time:   {adapt_time:.6f} s")
    print(f"Memory usage: current={adapt_cur_mem} bytes, peak={adapt_peak_mem} bytes")

    dt_minimum = min(dt_values)
    print(f"Minimum dt from the adaptive run = {dt_minimum:.3e}")

    # Optionally measure time/memory for a constant-dt run
    (final_state, const_time, const_cur_mem, const_peak_mem) = measure_time_and_memory(
        run_with_dt_minimum, dt_minimum
    )
    print("\n======== CONSTANT DT RUN ========")
    print(f"Using dt_minimum = {dt_minimum:e}")
    print(f"Total time:   {const_time:.6f} s")
    print(f"Memory usage: current={const_cur_mem} bytes, peak={const_peak_mem} bytes")
