# File: tang-polyq/src/tang_polyq/stochastic.py
import numpy as np
from dataclasses import dataclass
from .model import SBMLModel
from .scenarios import Scenario

@dataclass
class StochasticEnsembleResult:
    scenario_name: str
    n_cells: int
    tau: float
    t_hours: np.ndarray
    mean_y_alive: np.ndarray           # Shape: (25, N_out) - averaged over surviving cells only
    std_y_alive: np.ndarray            # Shape: (25, N_out) - SD across surviving cells
    mean_y_all: np.ndarray             # Shape: (25, N_out) - averaged over all cells (dead cells frozen)
    inclusion_freq_pct: np.ndarray     # % of cells with SeqAggP >= 1
    inclusion_freq_se: np.ndarray      # Binomial SE (%) = 100 * sqrt(p*(1-p)/N)
    cumulative_death_pct: np.ndarray   # % of cells dead by time t
    cumulative_death_se: np.ndarray    # Binomial SE (%) = 100 * sqrt(p*(1-p)/N)
    nucleation_times_h: np.ndarray     # Per-cell first passage time for SeqAggP >= 1

    @property
    def inclusion_freq_std(self) -> np.ndarray:
        return self.inclusion_freq_se

    @property
    def cumulative_death_std(self) -> np.ndarray:
        return self.cumulative_death_se

def run_tau_leaping_ensemble(
    model: SBMLModel,
    scenario: Scenario,
    n_cells: int = 300,
    t_end: float = 48.0 * 3600.0,
    tau: float = 2.0,
    save_every_s: float = 300.0,
    seed: int = 42,
) -> StochasticEnsembleResult:
    """
    Runs a vectorized Poisson tau-leaping ensemble across `n_cells` (default 300,
    matching Tang et al., 2010) in a single pass (<15 MB RAM).
    Uncertainty bands are computed via binomial standard error: 100 * sqrt(p*(1-p)/n_cells).
    Note on species means: `mean_y_alive` averages over surviving cells at time t
    (matching live-cell fluorescence microscopy for mRFPu), whereas `mean_y_all` includes
    dead cells frozen at their time of death.
    """
    rng = np.random.default_rng(seed)
    n_sp = len(model.species_ids)
    n_steps = int(np.round(t_end / tau))
    stride = max(1, int(np.round(save_every_s / tau)))
    n_out = n_steps // stride + 1

    t_out = np.zeros(n_out, dtype=np.float64)
    mean_y_alive = np.zeros((n_sp, n_out), dtype=np.float64)
    std_y_alive = np.zeros((n_sp, n_out), dtype=np.float64)
    mean_y_all = np.zeros((n_sp, n_out), dtype=np.float64)
    inc_pct = np.zeros(n_out, dtype=np.float64)
    death_pct = np.zeros(n_out, dtype=np.float64)

    y0 = scenario.get_initial_state(model)
    Y = np.repeat(y0[:, None], n_cells, axis=1).astype(np.float64)

    mean_y_alive[:, 0] = np.mean(Y, axis=1)
    std_y_alive[:, 0] = np.std(Y, axis=1)
    mean_y_all[:, 0] = np.mean(Y, axis=1)

    seq_idx = model.species_map["SeqAggP"]
    pi_idx = model.species_map["PIdeath"]
    p38_idx = model.species_map["p38death"]

    nucleation_times_h = np.full(n_cells, np.nan, dtype=np.float64)
    alive_mask = np.ones(n_cells, dtype=bool)

    p_pre = scenario.get_params_at(model, 0.0)
    p_post = scenario.get_params_at(model, t_end + 1.0)
    t_switch = scenario.switch_time
    N_mat = model.stoich_matrix

    out_idx = 1
    for step in range(1, n_steps + 1):
        t_curr = (step - 1) * tau
        p = p_post if (t_switch is not None and t_curr >= t_switch) else p_pre

        a = model.propensities(t_curr, Y, p)
        a[:, ~alive_mask] = 0.0

        lam = np.minimum(a * tau, 1e7)
        k_fires = rng.poisson(lam).astype(np.float64)

        Y = np.maximum(np.round(Y + (N_mat @ k_fires)), 0.0)

        newly_nucleated = np.isnan(nucleation_times_h) & (Y[seq_idx, :] >= 1.0)
        nucleation_times_h[newly_nucleated] = (step * tau) / 3600.0

        dead_now = (Y[pi_idx, :] >= 1.0) | (Y[p38_idx, :] >= 1.0)
        alive_mask = ~dead_now

        if step % stride == 0 and out_idx < n_out:
            t_out[out_idx] = step * tau
            mean_y_all[:, out_idx] = np.mean(Y, axis=1)
            if np.any(alive_mask):
                mean_y_alive[:, out_idx] = np.mean(Y[:, alive_mask], axis=1)
                std_y_alive[:, out_idx] = np.std(Y[:, alive_mask], axis=1)
            else:
                mean_y_alive[:, out_idx] = mean_y_all[:, out_idx]

            inc_pct[out_idx] = np.mean(Y[seq_idx, :] >= 1.0) * 100.0
            death_pct[out_idx] = np.mean(dead_now) * 100.0
            out_idx += 1

    p_inc = inc_pct / 100.0
    inc_se = 100.0 * np.sqrt(np.maximum(p_inc * (1.0 - p_inc), 0.0) / n_cells)

    p_death = death_pct / 100.0
    death_se = 100.0 * np.sqrt(np.maximum(p_death * (1.0 - p_death), 0.0) / n_cells)

    return StochasticEnsembleResult(
        scenario_name=scenario.name,
        n_cells=n_cells,
        tau=tau,
        t_hours=t_out / 3600.0,
        mean_y_alive=mean_y_alive,
        std_y_alive=std_y_alive,
        mean_y_all=mean_y_all,
        inclusion_freq_pct=inc_pct,
        inclusion_freq_se=inc_se,
        cumulative_death_pct=death_pct,
        cumulative_death_se=death_se,
        nucleation_times_h=nucleation_times_h,
    )