import numpy as np
import cantera as ct

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
    dt_old: float,
    main_method: str,
    x_main: np.ndarray,
    x_ref: np.ndarray,
    tolerance: float = 1e-12,
    gamma: float = 0.9,
    norm_type: int = 2) -> float:

    p_main: int = METHOD_ORDERS[main_method] # order of the main scheme

    error: float = np.linalg.norm(x_main[:-1] - x_ref[:-1], ord=norm_type) # computing errors for Y values alone (temperature ignored)
    # print("Error: ",error)
        
    # dt_new: float = gamma * ((dt_old**(p_main + 1) * tolerance) / error) ** (1.0 / (p_main + 1))
    dt_new: float = gamma * dt_old * (tolerance / error) ** (1.0 / (p_main + 1))
    return dt_new
