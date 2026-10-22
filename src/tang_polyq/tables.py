# File: tang-polyq/src/tang_polyq/tables.py
import os
import csv
import numpy as np
from typing import Dict, List
from .analysis import DeterministicMetrics
from .stability import StepSizeTrial
from .stochastic import StochasticEnsembleResult

def save_tables(
    metrics_by_scen: Dict[str, DeterministicMetrics],
    step_trials: List[StepSizeTrial],
    mc_results: Dict[str, StochasticEnsembleResult] = None,
    tau_sweep: Dict[float, StochasticEnsembleResult] = None,
    out_dir: str = "outputs/tables",
):
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "scenario_metrics_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Inclusion_Seed_Time_h", "Proteasome_Exhaustion_Time_h",
            "PolyQ_Oligomers_30h", "MisP_Oligomers_30h", "Death_at_30h_pct", "Death_at_48h_pct"
        ])
        for name, m in metrics_by_scen.items():
            idx30 = int(np.argmin(np.abs(m.t_hours - 30.0)))
            seed = f"{m.t_inclusion_seed_h:.2f}" if not np.isnan(m.t_inclusion_seed_h) else "N/A"
            exhaust = f"{m.t_proteasome_exhaust_h:.2f}" if not np.isnan(m.t_proteasome_exhaust_h) else "N/A"
            writer.writerow([
                name, seed, exhaust,
                f"{m.polyq_oligomers[idx30]:.1f}",
                f"{m.misp_oligomers[idx30]:.1f}",
                f"{m.death_at_30h_pct:.2f}",
                f"{m.death_at_48h_pct:.2f}",
            ])

    step_path = os.path.join(out_dir, "step_size_stability_study.csv")
    with open(step_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Sim_Horizon_h", "Step_Size_dt_s", "Is_Stable",
            "Rel_Error_SeqAggP", "Rel_Error_ROS", "Rel_Error_Death"
        ])
        for t in step_trials:
            stable_str = "Yes" if t.stable else "No"
            err_seq = f"{t.rel_error_SeqAggP:.4e}" if t.stable and not np.isnan(t.rel_error_SeqAggP) else "OVERFLOW"
            err_ros = f"{t.rel_error_ROS:.4e}" if t.stable and not np.isnan(t.rel_error_ROS) else "OVERFLOW"
            err_death = f"{t.rel_error_death:.4e}" if t.stable and not np.isnan(t.rel_error_death) else "OVERFLOW"
            writer.writerow([t.method, t.horizon_h, t.dt, stable_str, err_seq, err_ros, err_death])

    if mc_results:
        mc_path = os.path.join(out_dir, "stochastic_summary.csv")
        with open(mc_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Run_Label", "Scenario", "N_Cells", "Tau_s", "Median_Nucleation_h",
                "Inclusions_24h_pct", "Inclusions_30h_pct", "Inclusions_36h_pct",
                "Inclusions_42h_pct", "Inclusions_48h_pct",
                "Death_30h_pct", "Death_30h_SE", "Death_48h_pct", "Death_48h_SE"
            ])
            all_runs = list(mc_results.items())
            if tau_sweep:
                for tau_val, res in tau_sweep.items():
                    all_runs.append((f"untreated_tau_{tau_val:g}s", res))
            for label, mc in all_runs:
                def get_t(h):
                    return int(np.argmin(np.abs(mc.t_hours - h)))
                med_nuc = f"{np.nanmedian(mc.nucleation_times_h):.2f}" if np.any(~np.isnan(mc.nucleation_times_h)) else "N/A"
                i30, i48 = get_t(30.0), get_t(48.0)
                writer.writerow([
                    label, mc.scenario_name, mc.n_cells, mc.tau, med_nuc,
                    f"{mc.inclusion_freq_pct[get_t(24.0)]:.1f}",
                    f"{mc.inclusion_freq_pct[i30]:.1f}",
                    f"{mc.inclusion_freq_pct[get_t(36.0)]:.1f}",
                    f"{mc.inclusion_freq_pct[get_t(42.0)]:.1f}",
                    f"{mc.inclusion_freq_pct[i48]:.1f}",
                    f"{mc.cumulative_death_pct[i30]:.2f}",
                    f"{mc.cumulative_death_se[i30]:.2f}",
                    f"{mc.cumulative_death_pct[i48]:.2f}",
                    f"{mc.cumulative_death_se[i48]:.2f}",
                ])

    print(f"CSV Tables successfully generated in {out_dir}/")