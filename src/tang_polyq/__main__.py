# File: tang-polyq/src/tang_polyq/__main__.py
import numpy as np
from .model import load_default_model
from .scenarios import get_scenarios
from .solvers import solve_radau
from .stability import analyze_eigenvalue_evolution, run_step_size_study
from .analysis import compute_metrics, PAPER_TARGETS
from .stochastic import run_tau_leaping_ensemble
from .plotting import plot_all_figures
from .tables import save_tables

def main():
    print("=== Intracellular Huntington's Modeling Pipeline (v0.3 Final) ===")

    # 1. Load Model (Also verify dimer switch equivalence)
    model = load_default_model(use_mass_action_dimer=True)
    model_raw_sbml = load_default_model(use_mass_action_dimer=False)
    print(model.summary())

    scenarios = get_scenarios()
    metrics = {}

    # 2. Run Deterministic Scenarios
    print("\n--- Running Deterministic Scenarios (Radau) ---")
    for s_name, scen in scenarios.items():
        print(f"Solving {scen.label}...")
        res = solve_radau(model, scen)
        metrics[s_name] = compute_metrics(model, res)

    res_raw = solve_radau(model_raw_sbml, scenarios["untreated"])
    met_raw = compute_metrics(model_raw_sbml, res_raw)

    # 3. Step-Size and Stability Study
    print("\n--- Running Stability & Step-Size Study ---")
    untreated_res = solve_radau(model, scenarios["untreated"])
    eig_res = analyze_eigenvalue_evolution(model, untreated_res, scenarios["untreated"])
    trials = run_step_size_study(model, scenarios["untreated"])

    idx0 = int(np.argmin(np.abs(eig_res.t_hours - 0.0)))
    idx30 = int(np.argmin(np.abs(eig_res.t_hours - 30.0)))
    idx48 = int(np.argmin(np.abs(eig_res.t_hours - 48.0)))
    print(f"Max Eigenvalue at  0h: {eig_res.lambda_max_abs[idx0]:.4f} s^-1 (dt_max = {eig_res.dt_stability_limit[idx0]:.1f} s)")
    print(f"Max Eigenvalue at 30h: {eig_res.lambda_max_abs[idx30]:.4f} s^-1 (dt_max = {eig_res.dt_stability_limit[idx30]:.1f} s)")
    print(f"Max Eigenvalue at 48h: {eig_res.lambda_max_abs[idx48]:.4f} s^-1 (dt_max = {eig_res.dt_stability_limit[idx48]:.1f} s)")

    for H in [30.0, 48.0]:
        print(f"\nStep-Size Trial Results (T={int(H)}h):")
        for t in [tr for tr in trials if tr.horizon_h == H]:
            status = "STABLE" if t.stable else "UNSTABLE"
            err = f"{t.rel_error_death*100:.3e}%" if t.stable else "OVERFLOW"
            seq_err = f"{t.rel_error_SeqAggP*100:.3e}%" if t.stable else "OVERFLOW"
            print(f"  {t.method:5s} | dt={t.dt:4.1f}s | {status:8s} | Death Err: {err:10s} | SeqAggP Err: {seq_err}")

    # 4. Run Stochastic Ensembles (N = 300 cells) & Tau Sensitivity Sweep (tau = 1, 2, 5 s)
    print("\n--- Running Stochastic Ensembles (N = 300 cells) ---")
    print("  Running Untreated at tau = 1.0s, 2.0s, 5.0s (Tau sensitivity study)...")
    mc_unt_tau1 = run_tau_leaping_ensemble(model, scenarios["untreated"], n_cells=300, tau=1.0, seed=42)
    mc_unt_tau2 = run_tau_leaping_ensemble(model, scenarios["untreated"], n_cells=300, tau=2.0, seed=42)
    mc_unt_tau5 = run_tau_leaping_ensemble(model, scenarios["untreated"], n_cells=300, tau=5.0, seed=42)

    print("  Running PI at 24h (N = 300, tau = 2.0s)...")
    mc_pi = run_tau_leaping_ensemble(model, scenarios["pi_24h"], n_cells=300, tau=2.0, seed=101)

    print("  Running p38 Inhibitor (N = 300, tau = 2.0s)...")
    mc_p38 = run_tau_leaping_ensemble(model, scenarios["p38_inhibitor"], n_cells=300, tau=2.0, seed=202)

    mc_results = {
        "untreated": mc_unt_tau2,
        "pi_24h": mc_pi,
        "p38_inhibitor": mc_p38,
    }
    tau_sweep = {
        1.0: mc_unt_tau1,
        2.0: mc_unt_tau2,
        5.0: mc_unt_tau5,
    }

    # 5. Generate Figures and Tables
    print("\n--- Generating Figures & Tables ---")
    plot_all_figures(model, metrics, eig_res, trials, mc_results, tau_sweep)
    save_tables(metrics, trials, mc_results, tau_sweep)

    # 6. Final Analytical Audit
    print("\n=== THEORETICAL AUDIT & RESULTS ===")
    unt = metrics["untreated"]
    p38_inh = metrics["p38_inhibitor"]
    pi_24h = metrics["pi_24h"]
    no_fb_u = metrics["no_p38_ros_feedback"]
    no_fb_pi = metrics["pi_24h_no_feedback"]
    no_fb_p38 = metrics["p38_inh_no_feedback"]
    no_fb_bas = metrics["no_basal_ros_gen"]
    ctrl = metrics["no_polyq_control"]
    ctrl_noagg = metrics["no_polyq_no_agg"]

    idx30_m = int(np.argmin(np.abs(unt.t_hours - 30.0)))
    print("1. Oligomer Pools & Dimer Switch Verification:")
    print(f"   PolyQ Oligomers at 30h (AggPolyQ1..5): {unt.polyq_oligomers[idx30_m]:.1f} copies")
    print(f"   MisP Oligomers at 30h  (AggP1..5)    : {unt.misp_oligomers[idx30_m]:.1f} copies")
    print(f"   48h Untreated Death (Mass-Action X^2 vs Raw SBML X*(X-1)): {unt.death_at_48h_pct:.2f}% vs {met_raw.death_at_48h_pct:.2f}%")

    print("\n2. ODE Replication (30h & 48h Death):")
    print(f"   Untreated                 : 30h = {unt.death_at_30h_pct:.2f}% (Paper Fig 5A: ~21.0%), 48h = {unt.death_at_48h_pct:.2f}%")
    print(f"   p38 Blocker (kp38act=0.05): 30h = {p38_inh.death_at_30h_pct:.2f}% (Paper Fig 5A: ~8.0%),  48h = {p38_inh.death_at_48h_pct:.2f}%")

    print("\n3. p38 -> ROS Feedback Loop (Fig 7A Audit at 30h):")
    print(f"   No Feedback Untreated (kgenROSp38=0)  : {no_fb_u.death_at_30h_pct:.2f}% (Paper Fig 7A: ~6.7%)")
    print(f"   No Feedback + PI 24h  (kgenROSp38=0)  : {no_fb_pi.death_at_30h_pct:.2f}% (Paper Fig 7A: ~9.8%)")
    print(f"   No Feedback + p38 Inh (kgenROSp38=0)  : {no_fb_p38.death_at_30h_pct:.2f}% (Paper Fig 7A: ~4.7%)")
    print(f"   No Basal ROS          (kgenROS=0)     : {no_fb_bas.death_at_30h_pct:.2f}%")

    print("\n4. The Proteasome Inhibition (PI) Artifact & No-PolyQ Control Drift:")
    print(f"   Untreated ODE Proteasome Exhaustion   : {unt.t_proteasome_exhaust_h:.2f} h (48h Death Unt={unt.death_at_48h_pct:.2f}% vs PI={pi_24h.death_at_48h_pct:.2f}%)")
    print(f"   No-PolyQ Control Proteasome Exhaustion: {ctrl.t_proteasome_exhaust_h:.2f} h (48h ROS={ctrl.ros[-1]:.1f}, mRFPu={ctrl.mRFPu[-1]:.1f})")
    print(f"   No-PolyQ + kaggMisP=0 (48h Stable)    : Exhaustion=Never, 48h ROS={ctrl_noagg.ros[-1]:.1f}, mRFPu={ctrl_noagg.mRFPu[-1]:.1f}, FreeProt={ctrl_noagg.free_proteasome[-1]:.1f}")

    def mc_at(mc_obj, h):
        idx = int(np.argmin(np.abs(mc_obj.t_hours - h)))
        return mc_obj.inclusion_freq_pct[idx], mc_obj.inclusion_freq_se[idx], mc_obj.cumulative_death_pct[idx], mc_obj.cumulative_death_se[idx]

    print("\n5. Stochastic Ensemble (N=300, Binomial SE) & Tau-Sensitivity Audit:")
    for tau_v, res_t in sorted(tau_sweep.items()):
        inc30, se30, d30, dse30 = mc_at(res_t, 30.0)
        inc42, se42, _, _ = mc_at(res_t, 42.0)
        inc48, se48, d48, dse48 = mc_at(res_t, 48.0)
        med_n = np.nanmedian(res_t.nucleation_times_h)
        print(f"   [Untreated tau={tau_v:g}s] MedSeed={med_n:.2f}h | Inc 30/42/48h: {inc30:.1f}% / {inc42:.1f}% / {inc48:.1f}% (+/-{se48:.1f}%) | 30h Death: {d30:.1f}% (+/-{dse30:.1f}%)")

    inc30_pi, se30_pi, d30_pi, dse30_pi = mc_at(mc_pi, 30.0)
    inc48_pi, se48_pi, d48_pi, dse48_pi = mc_at(mc_pi, 48.0)
    _, _, d30_p38, dse30_p38 = mc_at(mc_p38, 30.0)
    print(f"   [PI at 24h tau=2s] Inc 30/48h: {inc30_pi:.1f}% / {inc48_pi:.1f}% (Paper: 26% / 77%) | 30h Death: {d30_pi:.1f}% +/- {dse30_pi:.1f}% (Paper: ~18%)")
    print(f"   [p38 Inh   tau=2s] 30h Death : {d30_p38:.1f}% +/- {dse30_p38:.1f}% (Paper Fig 5A: ~8.0%)")
    print("===================================================\n")

if __name__ == "__main__":
    main()