# File: tang-polyq/src/tang_polyq/plotting.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List
from .model import SBMLModel
from .analysis import DeterministicMetrics, PAPER_TARGETS
from .stability import EigenSpectrumResult, StepSizeTrial
from .stochastic import StochasticEnsembleResult

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

def plot_all_figures(
    model: SBMLModel,
    metrics_by_scen: Dict[str, DeterministicMetrics],
    eig_res: EigenSpectrumResult,
    step_trials: List[StepSizeTrial],
    mc_results: Dict[str, StochasticEnsembleResult],
    tau_sweep: Dict[float, StochasticEnsembleResult] = None,
    out_dir: str = "outputs/figures",
):
    os.makedirs(out_dir, exist_ok=True)
    plot_fig1_untreated_dynamics(metrics_by_scen["untreated"], out_dir)
    plot_fig2_treatment_comparisons(metrics_by_scen, out_dir)
    plot_fig3_feedback_loop_fig7(metrics_by_scen, out_dir)
    plot_fig4_stability_and_eigenvalues(eig_res, step_trials, out_dir)
    plot_fig5_control_drift_audit(metrics_by_scen, out_dir)
    plot_fig6_stochastic_vs_deterministic(model, metrics_by_scen, mc_results, tau_sweep, out_dir)
    print(f"All 6 figures saved to {out_dir}/")

def plot_fig1_untreated_dynamics(m: DeterministicMetrics, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = m.t_hours

    ax = axes[0, 0]
    ax.plot(t, m.inclusion_size, color="#1f77b4", lw=2, label="Sequestered Inclusion (SeqAggP)")
    ax.axvline(m.t_inclusion_seed_h, color="red", ls=":", label=f"Seed (SeqAggP>=1) at {m.t_inclusion_seed_h:.2f} h")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Inclusion Copy Number (SeqAggP)", color="#1f77b4")
    ax_olig = ax.twinx()
    ax_olig.plot(t, m.polyq_oligomers, color="#ff7f0e", lw=2, ls="--", label="PolyQ Oligomers (AggPolyQ1..5)")
    ax_olig.plot(t, m.misp_oligomers, color="#d62728", lw=1.8, ls="-.", label="MisP Oligomers (AggP1..5)")
    ax_olig.set_ylabel("Soluble Oligomer Copies", color="#ff7f0e")
    ax_olig.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_olig.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper left")
    ax.set_title("A. PolyQ/MisP Oligomer Plateau & Inclusion Growth")

    ax = axes[0, 1]
    ax.plot(t, m.free_proteasome, color="#2ca02c", lw=2, label="Free Proteasome")
    ax.plot(t, m.sequestered_proteasome, color="#d62728", lw=2, label="Bound Proteasome Complexes")
    ax.axvline(m.t_proteasome_exhaust_h, color="black", ls=":", label=f"Exhausted (<1 copy) at {m.t_proteasome_exhaust_h:.2f} h")
    ax.set_title("B. Untreated Proteasome Depletion")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Copies")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, m.ros, color="#9467bd", lw=2, label="ROS")
    ax2 = ax.twinx()
    ax2.plot(t, m.p38_p_frac * 100.0, color="#8c564b", lw=2, ls="--", label="Active p38_P (%)")
    ax2.grid(False)
    ax.set_title("C. ROS Amplification & p38 MAPK Activation")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("ROS Copies", color="#9467bd")
    ax2.set_ylabel("p38_P (%)", color="#8c564b")

    ax = axes[1, 1]
    ax.plot(t, m.death_pct, color="black", lw=2.5, label="Total Cumulative Death (%)")
    ax.plot(t, m.death_PI_only_pct, color="#e377c2", lw=1.8, ls="--", label="PI Hazard Route")
    ax.plot(t, m.death_p38_only_pct, color="#17becf", lw=1.8, ls="-.", label="p38 Hazard Route")
    ax.set_title("D. Hazard Cell Death Routes")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Death (%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_untreated_dynamics.png"))
    plt.close(fig)

def plot_fig2_treatment_comparisons(metrics: Dict[str, DeterministicMetrics], out_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

    scens = {
        "untreated": ("Untreated", "black", "-"),
        "pi_24h": ("PI at 24h (kproteff=0.05)", "red", "-."),
        "p38_inhibitor": ("p38 Inhibitor (kp38act=0.05)", "blue", "--"),
        "pi_24h_p38_inhibitor": ("PI at 24h + p38 Inhibitor", "purple", ":"),
    }

    for key, (label, col, ls) in scens.items():
        if key in metrics:
            m = metrics[key]
            ax.plot(m.t_hours, m.death_pct, color=col, ls=ls, lw=2.2, label=f"{label} (ODE)")
            target = PAPER_TARGETS["fig5A_death_30h_pct"].get(key)
            if target is not None:
                ax.plot(30.0, target, marker="X", color=col, markersize=10, ls="None",
                        label=f"{label} (Paper Fig 5A: {target:.0f}%)")

    ax.axvline(24.0, color="gray", ls=":", alpha=0.7, label="PI Addition Time (24 h)")
    ax.set_title("Pharmacological Interventions on Cell Death (Comparing to Fig 5A)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Cell Death (%)")
    ax.set_xlim(0, 48)
    ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_treatment_death.png"))
    plt.close(fig)

def plot_fig3_feedback_loop_fig7(metrics: Dict[str, DeterministicMetrics], out_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    m_u = metrics.get("untreated")
    if m_u:
        ax.plot(m_u.t_hours, m_u.death_pct, color="black", lw=2.2,
                label=f"With Feedback: Untreated ODE ({m_u.death_at_30h_pct:.2f}% at 30h)")
        ax.plot(30.0, PAPER_TARGETS["fig5A_death_30h_pct"]["untreated"], marker="o", color="black",
                markersize=9, ls="None", label="Paper Fig 5A Untreated (With Feedback: 21.0%)")

    fig7_scens = [
        ("no_p38_ros_feedback", "No Feedback: Untreated (kgenROSp38=0)", "red", "--", "X"),
        ("pi_24h_no_feedback", "No Feedback: PI at 24h (kgenROSp38=0)", "purple", ":", "^"),
        ("p38_inh_no_feedback", "No Feedback: p38 Inh (kgenROSp38=0)", "blue", "--", "s"),
    ]
    for key, label, col, ls, mk in fig7_scens:
        m = metrics.get(key)
        if m:
            ax.plot(m.t_hours, m.death_pct, color=col, ls=ls, lw=2.0,
                    label=f"{label} (ODE: {m.death_at_30h_pct:.2f}% at 30h)")
            t_val = PAPER_TARGETS["fig7A_death_30h_pct"].get(key)
            if t_val is not None:
                ax.plot(30.0, t_val, marker=mk, color=col, markersize=9, ls="None",
                        label=f"Paper Fig 7A Target ({t_val:.1f}%)")

    m_nf2 = metrics.get("no_basal_ros_gen")
    if m_nf2:
        ax.plot(m_nf2.t_hours, m_nf2.death_pct, color="#ff7f0e", ls="-.", lw=1.8,
                label=f"No Basal ROS (kgenROS=0, ODE: {m_nf2.death_at_30h_pct:.2f}% at 30h)")

    ax.set_title("Effect of Removing p38 -> ROS Feedback Across Treatments (Fig 5A vs. Fig 7A)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Cell Death (%)")
    ax.set_xlim(0, 48)
    ax.legend(fontsize=7.8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_feedback_loop_fig7.png"))
    plt.close(fig)

def plot_fig4_stability_and_eigenvalues(eig: EigenSpectrumResult, trials: List[StepSizeTrial], out_dir: str):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    ax1.plot(eig.t_hours, eig.lambda_max_abs, "k-", lw=2, label=r"Max $|\lambda_{\max}|$ (Spectral Radius)")
    ax1.plot(eig.t_hours, eig.lambda_min_active_abs, "b--", lw=1.6, alpha=0.7, label=r"Min Active $|\lambda|$ (FD floor post-20h)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Eigenvalue Magnitude (1/s)")
    ax1.set_title("A. Jacobian Eigenvalue Spectrum")
    ax1.legend(fontsize=8)

    ax2.plot(eig.t_hours, eig.dt_stability_limit, "r-", lw=2.2, label=r"$dt_{\max} = 2 / |\lambda_{\max}|$")
    ax2.axhline(17.8, color="gray", ls="--", alpha=0.7, label="30h Limit (~17.8 s)")
    ax2.axhline(6.56, color="black", ls=":", alpha=0.7, label="48h Limit (~6.56 s)")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel(r"Max Stable Explicit Step $dt_{\max}$ (s)")
    ax2.set_title("B. Explicit Stability Ceiling")
    ax2.legend(fontsize=8)

    for method, col, marker in [("Euler", "#1f77b4", "o"), ("RK2", "#2ca02c", "s")]:
        for H, ls in [(30.0, "-"), (48.0, "--")]:
            sub = [t for t in trials if t.method == method and t.horizon_h == H and t.stable]
            if sub:
                dts = [t.dt for t in sub]
                errs = [t.rel_error_death * 100.0 for t in sub]
                ax3.plot(dts, errs, color=col, marker=marker, ls=ls, lw=1.8,
                         label=f"{method} (T={int(H)}h)")

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Step Size $dt$ (s)")
    ax3.set_ylabel("Relative Death Error vs. Radau (%)")
    ax3.set_title("C. Step-Size Convergence ($O(dt)$ vs $O(dt^2)$)")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_stability_limits.png"))
    plt.close(fig)

def plot_fig5_control_drift_audit(metrics: Dict[str, DeterministicMetrics], out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m_ctrl = metrics.get("no_polyq_control")
    m_noagg = metrics.get("no_polyq_no_agg")

    exh_h = m_ctrl.t_proteasome_exhaust_h if m_ctrl else 39.62

    if m_ctrl:
        axes[0].plot(m_ctrl.t_hours, m_ctrl.mRFPu, "b-", lw=2, label="No-PolyQ Control (Published Params)")
        axes[0].axvline(exh_h, color="purple", ls=":", label=f"Free Proteasome < 1 at {exh_h:.2f} h")
        axes[1].plot(m_ctrl.t_hours, m_ctrl.ros, "r-", lw=2, label="No-PolyQ Control (Published Params)")
    if m_noagg:
        axes[0].plot(m_noagg.t_hours, m_noagg.mRFPu, "g--", lw=2, label="No-PolyQ + kaggMisP=0 (Stable ~366)")
        axes[1].plot(m_noagg.t_hours, m_noagg.ros, "k--", lw=2, label="No-PolyQ + kaggMisP=0 (Settled ~41)")

    axes[0].set_title(f"A. Control Drift: Free Proteasome Exhausted at {exh_h:.2f}h")
    axes[0].set_ylabel("mRFPu Reporter Copies")
    axes[0].set_xlabel("Time (hours)")
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("B. Control ROS: Climbs to ~281 (or Settles at ~41 if kaggMisP=0)")
    axes[1].set_ylabel("ROS Copies")
    axes[1].set_xlabel("Time (hours)")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_control_drift.png"))
    plt.close(fig)

def plot_fig6_stochastic_vs_deterministic(
    model: SBMLModel,
    metrics: Dict[str, DeterministicMetrics],
    mc_results: Dict[str, StochasticEnsembleResult],
    tau_sweep: Dict[float, StochasticEnsembleResult],
    out_dir: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    m_det = metrics["untreated"]
    mc_unt = mc_results.get("untreated")
    mc_pi = mc_results.get("pi_24h")
    mc_p38 = mc_results.get("p38_inhibitor")

    # Panel A: Inclusion Formation Frequency (with Binomial SE)
    ax = axes[0]
    if mc_unt is not None:
        t = mc_unt.t_hours
        mean_inc = mc_unt.inclusion_freq_pct
        se_inc = mc_unt.inclusion_freq_se
        ax.plot(t, mean_inc, "b-", lw=2, label=f"Stochastic Untreated (N={mc_unt.n_cells})")
        ax.fill_between(t, np.maximum(mean_inc - se_inc, 0), np.minimum(mean_inc + se_inc, 100), color="b", alpha=0.2)

    if mc_pi is not None:
        t_p = mc_pi.t_hours
        mean_p = mc_pi.inclusion_freq_pct
        se_p = mc_pi.inclusion_freq_se
        ax.plot(t_p, mean_p, "r-.", lw=2, label=f"Stochastic PI at 24h (N={mc_pi.n_cells})")
        ax.fill_between(t_p, np.maximum(mean_p - se_p, 0), np.minimum(mean_p + se_p, 100), color="r", alpha=0.15)

    ax.axvline(m_det.t_inclusion_seed_h, color="black", ls=":", lw=2,
               label=f"Deterministic ODE Seed ({m_det.t_inclusion_seed_h:.1f}h)")

    paper_x = list(PAPER_TARGETS["fig5B_inclusion_freq_pct"].keys())
    paper_u = [PAPER_TARGETS["fig5B_inclusion_freq_pct"][x]["untreated"] for x in paper_x]
    paper_pi = [PAPER_TARGETS["fig5B_inclusion_freq_pct"][x]["pi_24h"] for x in paper_x]
    ax.plot(paper_x, paper_u, "bo", markersize=7, ls="None", label="Paper Fig 5B (Untreated)")
    ax.plot(paper_x, paper_pi, "rX", markersize=8, ls="None", label="Paper Fig 5B (PI at 24h)")

    ax.set_title("A. Inclusion Frequency (+/- 1 Binomial SE)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("% of Cells with Inclusions")
    ax.legend(fontsize=7.5, loc="upper left")

    # Panel B: Population Cell Death (Untreated, PI at 24h, p38 Inhibitor)
    ax = axes[1]
    if mc_unt is not None:
        t = mc_unt.t_hours
        md = mc_unt.cumulative_death_pct
        se = mc_unt.cumulative_death_se
        ax.plot(t, md, "b-", lw=2, label="Stochastic Untreated (+/- SE)")
        ax.fill_between(t, np.maximum(md - se, 0), np.minimum(md + se, 100), color="b", alpha=0.2)

    if mc_pi is not None:
        t_p = mc_pi.t_hours
        md_p = mc_pi.cumulative_death_pct
        se_p = mc_pi.cumulative_death_se
        ax.plot(t_p, md_p, "r-.", lw=2, label="Stochastic PI at 24h (+/- SE)")
        ax.fill_between(t_p, np.maximum(md_p - se_p, 0), np.minimum(md_p + se_p, 100), color="r", alpha=0.15)

    if mc_p38 is not None:
        t_b = mc_p38.t_hours
        md_b = mc_p38.cumulative_death_pct
        se_b = mc_p38.cumulative_death_se
        ax.plot(t_b, md_b, "g--", lw=2, label="Stochastic p38 Inh (+/- SE)")
        ax.fill_between(t_b, np.maximum(md_b - se_b, 0), np.minimum(md_b + se_b, 100), color="g", alpha=0.2)

    ax.plot(m_det.t_hours, m_det.death_pct, "k:", lw=2, label="Deterministic ODE (Untreated & PI)")
    ax.plot(30.0, PAPER_TARGETS["fig5A_death_30h_pct"]["untreated"], "bo", markersize=8, ls="None", label="Paper Fig 5A Untreated (21%)")
    ax.plot(30.0, PAPER_TARGETS["fig5A_death_30h_pct"]["pi_24h"], "rX", markersize=9, ls="None", label="Paper Fig 5A PI at 24h (18%)")
    ax.plot(30.0, PAPER_TARGETS["fig5A_death_30h_pct"]["p38_inhibitor"], "gs", markersize=8, ls="None", label="Paper Fig 5A p38 Inh (8%)")

    ax.set_title("B. Cell Death: Stochastic vs. ODE (Fig 5A)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Cell Death (%)")
    ax.legend(fontsize=7.5, loc="upper left")

    # Panel C: Tau-Leaping Step Sensitivity (tau = 1, 2, 5 s)
    ax = axes[2]
    if tau_sweep:
        colors = {1.0: "#1f77b4", 2.0: "#2ca02c", 5.0: "#ff7f0e"}
        lstyles = {1.0: "-", 2.0: "--", 5.0: "-."}
        for tau_val, res in sorted(tau_sweep.items()):
            col = colors.get(tau_val, "black")
            ls = lstyles.get(tau_val, "-")
            ax.plot(res.t_hours, res.inclusion_freq_pct, color=col, ls=ls, lw=2,
                    label=f"tau = {tau_val:g} s (48h: {res.inclusion_freq_pct[-1]:.1f}%)")
            ax.fill_between(res.t_hours,
                            np.maximum(res.inclusion_freq_pct - res.inclusion_freq_se, 0),
                            np.minimum(res.inclusion_freq_pct + res.inclusion_freq_se, 100),
                            color=col, alpha=0.12)
    ax.plot(paper_x, paper_u, "ko", markersize=7, ls="None", label="Paper Fig 5B Target (Untreated)")
    ax.set_title("C. Tau-Leaping Sensitivity (Untreated, N=300)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("% of Cells with Inclusions")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_stochastic_comparison.png"))
    plt.close(fig)