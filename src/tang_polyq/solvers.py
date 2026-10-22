# File: tang-polyq/src/tang_polyq/solvers.py
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from .model import SBMLModel
from .scenarios import Scenario

@dataclass
class SimulationResult:
    solver_name: str
    scenario_name: str
    t: np.ndarray          # Shape: (N_out,) in seconds
    y: np.ndarray          # Shape: (25, N_out)
    dt: float
    success: bool
    message: str = ""

    @property
    def t_hours(self) -> np.ndarray:
        return self.t / 3600.0

    def get_species(self, model: SBMLModel, name: str) -> np.ndarray:
        return self.y[model.species_map[name], :]

def solve_euler(
    model: SBMLModel,
    scenario: Scenario,
    t_end: float = 48.0 * 3600.0,
    dt: float = 1.0,
    save_every_s: float = 60.0,
    blowup_threshold: float = 1e12,
) -> SimulationResult:
    y = scenario.get_initial_state(model)
    p_pre = scenario.get_params_at(model, 0.0)
    p_post = scenario.get_params_at(model, t_end + 1.0)
    t_switch = scenario.switch_time

    t_list = [0.0]
    y_list = [y.copy()]
    next_save_t = save_every_s

    t_curr = 0.0
    success = True
    msg = "Completed"

    while t_curr < t_end - 1e-12:
        dt_step = min(dt, t_end - t_curr)
        # Do not step across the 24h PI switch without hitting it exactly
        if t_switch is not None and t_curr < t_switch < t_curr + dt_step:
            dt_step = t_switch - t_curr

        p = p_post if (t_switch is not None and t_curr >= t_switch - 1e-12) else p_pre
        dy = model.rhs(t_curr, y, p)
        y = y + dt_step * dy
        t_curr += dt_step

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > blowup_threshold or np.min(y) < -1.0:
            success = False
            msg = f"Unstable blowup or negative copy number at t = {t_curr/3600.0:.2f} h"
            break

        if t_curr >= next_save_t - 1e-9 or abs(t_curr - t_end) <= 1e-9:
            t_list.append(t_curr)
            y_list.append(y.copy())
            next_save_t += save_every_s

    return SimulationResult(
        "Euler", scenario.name,
        np.array(t_list, dtype=np.float64),
        np.column_stack(y_list),
        dt, success, msg,
    )

def solve_rk2(
    model: SBMLModel,
    scenario: Scenario,
    t_end: float = 48.0 * 3600.0,
    dt: float = 1.0,
    save_every_s: float = 60.0,
    blowup_threshold: float = 1e12,
) -> SimulationResult:
    y = scenario.get_initial_state(model)
    p_pre = scenario.get_params_at(model, 0.0)
    p_post = scenario.get_params_at(model, t_end + 1.0)
    t_switch = scenario.switch_time

    t_list = [0.0]
    y_list = [y.copy()]
    next_save_t = save_every_s

    t_curr = 0.0
    success = True
    msg = "Completed"

    while t_curr < t_end - 1e-12:
        dt_step = min(dt, t_end - t_curr)
        if t_switch is not None and t_curr < t_switch < t_curr + dt_step:
            dt_step = t_switch - t_curr

        t_next = t_curr + dt_step
        p = p_post if (t_switch is not None and t_curr >= t_switch - 1e-12) else p_pre

        k1 = model.rhs(t_curr, y, p)
        y_pred = y + dt_step * k1
        k2 = model.rhs(t_next, y_pred, p)
        y = y + 0.5 * dt_step * (k1 + k2)
        t_curr = t_next

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > blowup_threshold or np.min(y) < -1.0:
            success = False
            msg = f"Unstable blowup or negative copy number at t = {t_curr/3600.0:.2f} h"
            break

        if t_curr >= next_save_t - 1e-9 or abs(t_curr - t_end) <= 1e-9:
            t_list.append(t_curr)
            y_list.append(y.copy())
            next_save_t += save_every_s

    return SimulationResult(
        "RK2", scenario.name,
        np.array(t_list, dtype=np.float64),
        np.column_stack(y_list),
        dt, success, msg,
    )

def solve_radau(
    model: SBMLModel,
    scenario: Scenario,
    t_end: float = 48.0 * 3600.0,
    save_every_s: float = 60.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> SimulationResult:
    y0 = scenario.get_initial_state(model)
    t_eval = np.arange(0.0, t_end + 0.5 * save_every_s, save_every_s)
    if abs(t_eval[-1] - t_end) > 1e-6:
        t_eval = np.append(t_eval, t_end)

    if scenario.switch_time is None or scenario.switch_time >= t_end or scenario.switch_time <= 0.0:
        p = scenario.get_params_at(model, 0.0)
        sol = solve_ivp(
            fun=lambda t, y: model.rhs(t, y, p),
            t_span=(0.0, t_end),
            y0=y0,
            method="Radau",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        return SimulationResult("Radau", scenario.name, sol.t, sol.y, save_every_s, sol.success, sol.message)
    else:
        t_sw = scenario.switch_time
        t_eval_1 = t_eval[t_eval <= t_sw]
        if t_eval_1[-1] < t_sw:
            t_eval_1 = np.append(t_eval_1, t_sw)
        p1 = scenario.get_params_at(model, 0.0)
        sol1 = solve_ivp(
            fun=lambda t, y: model.rhs(t, y, p1),
            t_span=(0.0, t_sw),
            y0=y0,
            method="Radau",
            t_eval=t_eval_1,
            rtol=rtol,
            atol=atol,
        )
        t_eval_2 = t_eval[t_eval > t_sw]
        t_eval_2_full = np.insert(t_eval_2, 0, t_sw)
        p2 = scenario.get_params_at(model, t_sw + 1.0)
        sol2 = solve_ivp(
            fun=lambda t, y: model.rhs(t, y, p2),
            t_span=(t_sw, t_end),
            y0=sol1.y[:, -1],
            method="Radau",
            t_eval=t_eval_2_full,
            rtol=rtol,
            atol=atol,
        )
        t_comb = np.concatenate([sol1.t, sol2.t[1:]])
        y_comb = np.hstack([sol1.y, sol2.y[:, 1:]])
        return SimulationResult("Radau", scenario.name, t_comb, y_comb, save_every_s, sol1.success and sol2.success, sol2.message)