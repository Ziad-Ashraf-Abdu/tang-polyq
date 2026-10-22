# File: tang-polyq/src/tang_polyq/stability.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from .model import SBMLModel
from .scenarios import Scenario
from .solvers import solve_radau, solve_euler, solve_rk2, SimulationResult

def compute_jacobian(model: SBMLModel, y: np.ndarray, params: Dict[str, float], eps: float = 1e-7) -> np.ndarray:
    n = len(y)
    J = np.zeros((n, n), dtype=np.float64)
    f0 = model.rhs(0.0, y, params)
    for j in range(n):
        step = eps * max(1.0, abs(y[j]))
        y_pert = y.copy()
        y_pert[j] += step
        f1 = model.rhs(0.0, y_pert, params)
        J[:, j] = (f1 - f0) / step
    return J

@dataclass
class EigenSpectrumResult:
    t_hours: np.ndarray
    lambda_max_abs: np.ndarray
    lambda_min_active_abs: np.ndarray
    stiffness_ratio: np.ndarray
    dt_stability_limit: np.ndarray

def analyze_eigenvalue_evolution(
    model: SBMLModel,
    radau_res: SimulationResult,
    scenario: Scenario,
    sample_hours: List[float] = None,
) -> EigenSpectrumResult:
    if sample_hours is None:
        sample_hours = list(np.linspace(0.0, 48.0, 49))

    exclude_names = {"PIdeath", "p38death", "p38"}
    base_dynamic_mask = np.array([s not in exclude_names for s in model.species_ids], dtype=bool)

    t_h_list = []
    lmax_list = []
    lmin_list = []
    stiff_list = []
    dt_lim_list = []

    for h in sample_hours:
        idx = int(np.argmin(np.abs(radau_res.t_hours - h)))
        t_sec = radau_res.t[idx]
        y_vec = radau_res.y[:, idx]
        p = scenario.get_params_at(model, t_sec)
        J = compute_jacobian(model, y_vec, p)

        eigs_full = np.linalg.eigvals(J)
        lmax = float(np.max(np.abs(eigs_full)))

        active_mask = base_dynamic_mask & (y_vec >= 0.5)
        J_sub = J[np.ix_(active_mask, active_mask)]
        eigs_sub = np.abs(np.linalg.eigvals(J_sub))
        valid_sub = eigs_sub[eigs_sub > max(1e-8, lmax * 1e-7)]
        lmin = float(np.min(valid_sub)) if len(valid_sub) > 0 else lmax

        t_h_list.append(radau_res.t_hours[idx])
        lmax_list.append(lmax)
        lmin_list.append(lmin)
        stiff_list.append(lmax / lmin if lmin > 0 else 1.0)
        dt_lim_list.append(2.0 / lmax if lmax > 0 else np.inf)

    return EigenSpectrumResult(
        t_hours=np.array(t_h_list),
        lambda_max_abs=np.array(lmax_list),
        lambda_min_active_abs=np.array(lmin_list),
        stiffness_ratio=np.array(stiff_list),
        dt_stability_limit=np.array(dt_lim_list),
    )

@dataclass
class StepSizeTrial:
    method: str
    horizon_h: float
    dt: float
    stable: bool
    rel_error_SeqAggP: float
    rel_error_ROS: float
    rel_error_death: float

def run_step_size_study(
    model: SBMLModel,
    scenario: Scenario,
    dts: List[float] = None,
    horizons_h: List[float] = None,
) -> List[StepSizeTrial]:
    if dts is None:
        dts = [1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    if horizons_h is None:
        horizons_h = [30.0, 48.0]

    ref = solve_radau(model, scenario, t_end=max(horizons_h) * 3600.0, save_every_s=60.0)
    trials = []

    for H in horizons_h:
        t_end = H * 3600.0
        ref_idx = int(np.argmin(np.abs(ref.t - t_end)))
        y_ref = ref.y[:, ref_idx]
        d_ref = 1.0 - np.exp(-(y_ref[model.species_map["PIdeath"]] + y_ref[model.species_map["p38death"]]))

        for dt in dts:
            for method_name, solver_fn in [("Euler", solve_euler), ("RK2", solve_rk2)]:
                res = solver_fn(model, scenario, t_end=t_end, dt=dt, save_every_s=600.0)
                if not res.success or abs(res.t[-1] - t_end) > 1e-5:
                    trials.append(StepSizeTrial(method_name, H, dt, False, np.nan, np.nan, np.nan))
                else:
                    y_fin = res.y[:, -1]
                    if np.any(y_fin < -1e-3) or not np.all(np.isfinite(y_fin)):
                        trials.append(StepSizeTrial(method_name, H, dt, False, np.nan, np.nan, np.nan))
                        continue
                    d_fin = 1.0 - np.exp(-(y_fin[model.species_map["PIdeath"]] + y_fin[model.species_map["p38death"]]))
                    err_seq = abs(y_fin[model.species_map["SeqAggP"]] - y_ref[model.species_map["SeqAggP"]]) / max(1.0, y_ref[model.species_map["SeqAggP"]])
                    err_ros = abs(y_fin[model.species_map["ROS"]] - y_ref[model.species_map["ROS"]]) / max(1.0, y_ref[model.species_map["ROS"]])
                    err_d = abs(d_fin - d_ref) / max(1e-8, d_ref)
                    trials.append(StepSizeTrial(method_name, H, dt, True, float(err_seq), float(err_ros), float(err_d)))

    return trials