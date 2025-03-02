import numpy as np
import cantera as ct
from typing import Callable

METHOD_ORDERS: dict[str, int] = {
    "rk1": 1,
    "rk2": 2,
    "rk2_heun": 2,
    "rk3": 3,
    "rk3_ssp": 3,
    "rk4": 4,
    "rk5": 5
}

def get_dt(
    tk: float,
    xk: np.ndarray,
    dt_old: float,
    func: Callable[[float, np.ndarray, ct.Solution, float], np.ndarray],
    main_method: str,
    ref_method: str,
    gas_obj: ct.Solution,
    dens: float,
    tolerance: float,
    gamma: float = 0.9,
    norm_type: int = 2) -> tuple[float]:


    solutions: np.ndarray = timestepper(
        tk=tk,
        xk=xk,
        dt=dt_old,
        func=func,
        required_methods=[main_method, ref_method],
        gas_obj=gas_obj,
        dens=dens)
    
    x_main: np.ndarray = solutions[0] #Main_scheme
    x_ref: np.ndarray = solutions[1] #reference_scheme

    p_main: int = METHOD_ORDERS[main_method] # order of the main scheme

    error: float = np.linalg.norm(x_main[:-1] - x_ref[:-1], ord=norm_type) # computing errors for Y values alone (temperature ignored)
    tolerance = 1e-30

    if error < 1e-30:
        error = 1e-30
        
    dt_new: float = gamma * ((dt_old**(p_main + 1) * tolerance) / error) ** (1.0 / (p_main + 1))
    return dt_new
