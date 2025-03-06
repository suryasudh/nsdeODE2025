import numpy as np
from typing import Callable
import cantera as ct

def rk_methods(method, dt, xk, f1, f2main, f2_o1, f2_o2,
               f3main, f3_o1, f3_o2, f3_o3,
               f4main, f4_o1, f5main, f6main):
        if method == "rk1" :
            return rk1(dt, xk, f1),
        if method == "rk2" :
            return rk2(dt, xk, f1, f2main),
        if method == "rk3" :
            return rk3(dt, xk, f1, f2main, f3_o1),
        if method == "rk4" :
            return rk4(dt, xk, f1, f2main, f3main, f4main),
        if method == "rk5" :
            return rk5(dt, xk, f1, f2_o2, f3_o3, f4_o1, f5main, f6main),
        if method == "rk2_heun" :
            return rk2_heun(dt, xk, f1, f2_o1),
        if method == "rk3_ssp" :
            return rk3_ssp(dt, xk, f1, f2_o1, f3_o2)


def rk5(dt, xk, f1, f2_o2, f3_o3, f4_o1, f5main, f6main):
    """
    """
    x_out = xk + ((dt/90) * ((7 * f1) + (0 * f2_o2) + (32 * f3_o3) + (12 * f4_o1) + (32 * f5main) + (7 * f6main)))
    return x_out


def rk4(dt, xk, f1, f2main, f3main, f4main):
    """
    """
    x_out = xk + ((dt/6) * (f1 + (2*f2main) + (2*f3main) + f4main))
    return x_out


def rk3(dt, xk, f1, f2main, f3_o1):
    """
    """
    x_out = xk + ((dt/6) * (f1 + (4*f2main) + f3_o1))
    return x_out


def rk2(dt, xk, f1, f2main):
    """
    """
    x_out = xk + ((dt) * (f2main))
    return x_out


def rk1(dt, xk, f1):
    """
    """
    x_out = xk + ((dt) * (f1))
    return x_out


def rk3_ssp(dt, xk, f1, f2_o1, f3_o2):
    """
    """
    x_out = xk + ((dt/6) * (((1/6) * f1) + ((1/6) * f2_o1) + ((4/6) * f3_o2)))
    return x_out


def rk2_heun(dt, xk, f1, f2_o1):
    """
    """
    x_out = xk + ((dt) * (((1/2) * f1) + ((1/2) * f2_o1)))
    return x_out

# TODO: include argument for specifying precision
def timestepper(tk:float, xk:np.ndarray, dt:float,
                func:Callable[[float, np.ndarray], np.ndarray],
                required_methods:list[str], gas_obj:ct.Solution, dens:float, prec:np.dtype) -> np.ndarray:
    slopes = {
        "rk1" : {"f1"},
        "rk2" : {"f1","f2main"},
        "rk3" : {"f1", "f2main", "f3_o1"},
        "rk4" : {"f1", "f2main", "f3main", "f4main"},
        "rk5" : {"f1", "f2_o2", "f3_o3", "f4_o1", "f5main", "f6main"},
        "rk2_heun" : {"f1", "f2_o1"},
        "rk3_ssp" : {"f1", "f2_o1", "f3_o2"}
    }

    required_slopes = set()
    for i in range(len(required_methods)):
        required_slopes = required_slopes.union(slopes[required_methods[i]])

    required_slopes = sorted(list(required_slopes))
    # print(required_slopes)

    f1, f2_o1, f2main, f3_o1, f3_o2, f3main, f4main = None, None, None, None, None, None, None
    f2_o2, f3_o3, f4_o1, f5main, f6main = None, None, None, None, None


    for reqdstr1 in required_slopes:
        if reqdstr1 == 'f1':
            f1 = func(tk, xk, gas_obj, dens, prec)

        if reqdstr1 == "f2_o1":
            f2_o1 = func((tk + (dt)), (xk + (1 * f1) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f2main":
            f2main = func((tk + (dt/2)), (xk + ((1/2) * f1) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f3main":
            f3main = func((tk + (dt/2)), (xk + ((1/2) * f2main) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f3_o1":
            f3_o1 = func((tk + (dt)), (xk + ((-1 * f1) + (2 * f2main)) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f3_o2":
            f3_o2 = func((tk + (dt/2)), (xk + (((1/4) * f1) + ((1/4) * f2_o1)) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f4main":
            f4main = func((tk + dt), (xk + (1 * f3main) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f2_o2":
            f2_o2 = func((tk + ((1/4) * dt)), (xk + ((1/4) * f1) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f3_o3":
            f3_o3 = func((tk + ((1/4) * dt)), (xk + (((1/8) * f1) + ((1/8) * f2_o2)) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f4_o1":
            f4_o1 = func((tk + ((1/2) * dt)), (xk + (((-1/2) * f2_o2) + ((1) * f3_o3)) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f5main":
            f5main = func((tk + ((3/4) * dt)), (xk + (((3/16) * f1) + ((9/16) * f4_o1)) * dt), gas_obj, dens, prec)

        if reqdstr1 == "f6main":
            f6main = func((tk + (1 * dt)), (xk + (((-3/7) * f1) +
                                                ((2/7) * f2_o2) +
                                                ((12/7) * f3_o3) +
                                                ((-12/7) * f4_o1) +
                                                ((8/7) * f5main)) * dt), gas_obj, dens, prec)
            

    required_outputs = []
    for _method in required_methods:
        required_outputs.append(rk_methods(method=_method, dt=dt, xk=xk, f1=f1,
                                           f2main=f2main, f2_o1=f2_o1, f2_o2=f2_o2,
                                           f3main=f3main, f3_o1=f3_o1, f3_o2=f3_o2, f3_o3=f3_o3,
                                           f4main=f4main, f4_o1=f4_o1, f5main=f5main, f6main=f6main))
    
    return np.array(required_outputs)


####################################################################
### Adams Bashforth Methods
####################################################################
# Similarly now, making functions related to
# Adam's Bashforth methods

def ab_methods(method, dt, xk, f1, f2, f3, f4, f5):
        if method == "ab1" :
            return ab1(dt, xk, f1)
        if method == "ab2" :
            return ab2(dt, xk, f1, f2)
        if method == "ab3" :
            return ab3(dt, xk, f1, f2, f3)
        if method == "ab4" :
            return ab4(dt, xk, f1, f2, f3, f4)
        if method == "ab5" :
            return ab5(dt, xk, f1, f2, f3, f4, f5)


def ab1(dt, xk, fn):
    """
    """
    x_out = xk + ((dt) * (fn))
    return x_out

def ab2(dt, xk, fn, fnm1):
    """
    """
    x_out = xk + ((dt/2) * ((3 * fn) - (1 * fnm1)))
    return x_out

def ab3(dt, xk, fn, fnm1, fnm2):
    """
    """
    x_out = xk + ((dt/12) * ((23 * fn) - (16 * fnm1) + (5 * fnm2)))
    return x_out

def ab4(dt, xk, fn, fnm1, fnm2, fnm3):
    """
    """
    x_out = xk + ((dt/24) * ((55 * fn) - (59 * fnm1) + (37 * fnm2) - (9 * fnm3)))
    return x_out

def ab5(dt, xk, fn, fnm1, fnm2, fnm3, fnm4):
    """
    """
    x_out = xk + ((dt/720) * ((1901 * fn) - (2774 * fnm1) + (2616 * fnm2) - (1274 * fnm3) + (251 * fnm4)))
    return x_out



# TODO: include argument for specifying precision
def timestepper_ab(tk:float, xk:np.ndarray, dt:float,
                func:Callable[[float, np.ndarray], np.ndarray],
                required_methods:list[str], gas_obj:ct.Solution, dens:float, prec) -> np.ndarray:
    slopes = {
        "ab1" : {"f1"},
        "ab2" : {"f1","f2"},
        "ab3" : {"f1", "f2", "f3"},
        "ab4" : {"f1", "f2", "f3", "f4"},
        "ab5" : {"f1", "f2", "f3", "f4", "f5"},
    }

    required_slopes = set()
    for i in range(len(required_methods)):
        required_slopes = required_slopes.union(slopes[required_methods[i]])

    required_slopes = sorted(list(required_slopes))
    # print(required_slopes)

    f1ab, f2ab, f3ab, f4ab, f5ab = None, None, None, None, None

    for reqdstr1 in required_slopes:
        if reqdstr1 == 'f1':
            f1ab = func(tk, xk[0], gas_obj, dens, prec)

        if reqdstr1 == "f2":
            f2ab = func((tk - (dt * 1)), (xk[1]), gas_obj, dens, prec)

        if reqdstr1 == "f3":
            f3ab = func((tk - (dt * 2)), (xk[2]), gas_obj, dens, prec)

        if reqdstr1 == "f4":
            f4ab = func((tk - (dt * 3)), (xk[3]), gas_obj, dens, prec)

        if reqdstr1 == "f5":
            f5ab = func((tk - (dt * 4)), (xk[4]), gas_obj, dens, prec)


    required_outputs = []
    for _method in required_methods:
        required_outputs.append(ab_methods(method=_method, dt=dt, xk=xk, f1=f1ab, f2=f2ab, f3=f3ab, f4=f4ab, f5=f5ab))

    return np.array(required_outputs)
















############################################################
### TESTING PART
############################################################
# def func_example1(tk, xk):
#     """ An example from
#     https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Active_Calculus_(Boelkins_et_al.)/07%3A_Differential_Equations/7.06%3A_Population_Growth_and_the_Logistic_Equation#:~:text=dPdt%3DkP(N%E2%88%92P),be%20sustained%20by%20the%20environment.&text=%E2%88%AB1P(N%E2%88%92P,P%3D%E2%88%ABk%20dt.
#     to test our code
#     """
#     return (0.002 * xk * (12.5 - xk))


# if __name__ == "__main__":
#     dt_step = 0.01
#     x_0_initial = 6.084

#     required_methods = ["rk1", "rk3", "rk4"]
#     x_sols = np.empty((int(200 * (1/dt_step)), len(required_methods)))
#     x_sols[:,:] = np.nan
    

#     x_old = 6.084

#     for i in range(1, len(x_sols)):
#         x_vals_new = timestepper(tk=0, xk=x_old, dt=dt_step, func=func_example1, required_methods=required_methods)
#         x_sols[i,:] = x_vals_new[:, 0]
#         x_old = x_vals_new[0, 0]
#         #print(x_vals_new.shape)
    

#     ###########################################
#     ### Plotting
#     ###########################################
#     fig, ax = plt.subplots(figsize=(10,6))
#     ax.plot(np.arange(0,200, dt_step), x_sols[:, 0], c="red", label=required_methods[0])
#     ax.plot(np.arange(0,200, dt_step), x_sols[:, 1], c="blue", label=required_methods[1])
#     ax.plot(np.arange(0,200, dt_step), x_sols[:, 2], c="green", label=required_methods[2])
    

#     ax.set(
#         title="example population growth problem",
#         xlim=(0,200),
#         ylim=(0,15),
#         yticks=(3,6,9,12,15),
#     )

#     ax.legend()
#     ax.grid(visible=True, which="both")
#     plt.show()