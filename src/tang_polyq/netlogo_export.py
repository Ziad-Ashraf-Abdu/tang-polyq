# File: tang-polyq/src/tang_polyq/netlogo_export.py
"""Run recorder and exporter for ``Tang_PolyQ_Interactive.nlogo``.

The NetLogo model calls three things from here (through the ``py`` extension):

* ``RunRecorder(...)``    created once in ``setup``
* ``rec.add(...)``        called once per tick (time series) and, for the tau-leaping
  ensemble, ``rec.add_cells(...)`` (per-cell first-inclusion and death times)
* ``save_run(rec, status)`` called when the run ends (48 h reached, solver blow-up) or
  when the "Save Data Now" button is pressed

Files written under ``outputs/NetLogo_Runs/`` (relative to the working directory, i.e. the
repository root when NetLogo is started from there):

    data/<run_id>_timeseries.csv   one row per tick
    data/<run_id>_cells.csv        one row per cell (tau-leaping runs only)
    data/<run_id>_meta.json        settings, effective parameters, summary metrics
    data/netlogo_run_summary.csv   one row appended per saved run (all runs together)
    figures/<run_id>.png           clean 300-dpi figure for the report

Re-draw every figure from the saved data (e.g. after changing the style):

    python -m tang_polyq.netlogo_export replot [outputs/NetLogo_Runs]
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np

DEFAULT_OUT_DIR = os.path.join("outputs", "NetLogo_Runs")
REPORT_HOURS = (24, 30, 36, 42, 48)
ENSEMBLE_SOLVER = "Tau-Leap (Ensemble)"
SUMMARY_FILE = "netlogo_run_summary.csv"

TS_COLUMNS = (
    "time_h", "death_pct", "death_pi_route_pct", "death_p38_route_pct", "inclusion_pct",
    "polyq_oligomers", "misp_oligomers", "seq_agg_p", "free_proteasome", "bound_proteasome",
    "ros", "mrfpu", "p38_active_pct", "lambda_max_per_s", "dt_max_s",
)
CELL_COLUMNS = ("cell_id", "inclusion_time_h", "death_time_h", "death_route")

_NAN = float("nan")


def _hours_cols():
    cols = []
    for h in REPORT_HOURS:
        cols += [f"death_{h}h_pct", f"death_{h}h_se_pct", f"inclusion_{h}h_pct", f"inclusion_{h}h_se_pct"]
    return cols


SUMMARY_COLUMNS = (
    ["run_id", "saved_at", "status", "scenario", "solver", "dt_s", "steps_per_tick", "n_cells", "seed",
     "mass_action_dimer", "custom_sliders", "kp38act", "kproteff_after_24h", "kgenROSp38", "kaggMisP",
     "t_end_h", "n_rows"]
    + _hours_cols()
    + ["t_seed_h", "t_proteasome_exhausted_h", "median_nucleation_h", "cells_with_inclusion_pct",
       "final_ros", "final_mrfpu"]
)

NOTES = [
    "ODE solvers (Euler/RK2): inclusion_pct is a 0/100 indicator (the mean-field cell either has "
    "SeqAggP >= 1 or not), not a frequency; death_pct is 1 - exp(-(PIdeath + p38death)).",
    "Tau-leaping: death_pct and inclusion_pct are fractions of all cells; the species columns "
    "(oligomers, proteasome, ROS, mRFPu, ...) are means over cells that are still alive.",
    "t_seed_h and t_proteasome_exhausted_h come from the plotted (mean) trajectory; in tau-leaping runs "
    "use median_nucleation_h for the per-cell inclusion time (resolution = one tick).",
    "SE columns are binomial standard errors sqrt(p(1-p)/N) (tau-leaping only).",
]


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        return "nan" if not math.isfinite(x) else format(float(x), ".10g")
    return str(x)


def _clean(obj):
    """Make an object JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def effective_params(p_start: Dict[str, float], p_after_switch: Dict[str, float],
                     custom: bool, sliders: Dict[str, float]) -> Dict[str, float]:
    """Parameter values a run actually uses (mirrors the override logic in the .nlogo ``go``)."""
    eff = {
        "kp38act": float(p_start["kp38act"]),
        "kproteff_after_24h": float(p_after_switch["kproteff"]),
        "kgenROSp38": float(p_start["kgenROSp38"]),
        "kaggMisP": float(p_start["kaggMisP"]),
    }
    if custom:
        for k in eff:
            if k in sliders:
                eff[k] = float(sliders[k])
    return eff


def _first_time(t: np.ndarray, y: np.ndarray, threshold: float, above: bool) -> float:
    idx = np.where(y >= threshold)[0] if above else np.where(y < threshold)[0]
    return float(t[idx[0]]) if idx.size else _NAN


# ----------------------------------------------------------------------------------------------
# recorder
# ----------------------------------------------------------------------------------------------
class RunRecorder:
    def __init__(self, scenario: str, solver: str, dt_s: float, steps_per_tick: int, n_cells: int,
                 seed: int, mass_action_dimer: bool, custom_sliders: bool,
                 params: Optional[Dict[str, float]] = None, out_dir: str = DEFAULT_OUT_DIR):
        self.scenario = str(scenario)
        self.solver = str(solver)
        self.is_ensemble = self.solver == ENSEMBLE_SOLVER
        self.dt_s = float(dt_s)
        self.steps_per_tick = int(steps_per_tick)
        self.n_cells = int(n_cells) if self.is_ensemble else 0
        self.seed = int(seed) if self.is_ensemble else None
        self.mass_action_dimer = bool(mass_action_dimer)
        self.custom_sliders = bool(custom_sliders)
        self.params = dict(params or {})
        self.out_dir = out_dir
        self.started = datetime.now()

        short = {"Euler (ODE)": "Euler", "RK2 (ODE)": "RK2", ENSEMBLE_SOLVER: "TauLeap"}.get(self.solver, "run")
        tag = f"__seed{self.seed}" if self.is_ensemble else ""
        self.run_id = f"{self.scenario}__{short}__dt{self.dt_s:g}s{tag}__{self.started:%Y%m%d-%H%M%S}"

        self.rows = []
        self.nucleation_h = np.full(self.n_cells, np.nan)
        self.death_h = np.full(self.n_cells, np.nan)
        self.death_route = np.zeros(self.n_cells, dtype=np.int8)  # 0 alive, 1 PI, 2 p38, 3 both
        self.paths: Dict[str, str] = {}

    # -- recording ---------------------------------------------------------------------------
    def add(self, t_h, death_pct, pi_pct, p38_pct, inclusion_pct, polyq_olig, misp_olig, seq_agg,
            free_prot, bound_prot, ros, mrfpu, p38_active_pct, lambda_max=0.0, dt_max=0.0):
        row = [float(t_h), float(death_pct), float(pi_pct), float(p38_pct), float(inclusion_pct),
               float(polyq_olig), float(misp_olig), float(seq_agg), float(free_prot), float(bound_prot),
               float(ros), float(mrfpu), float(p38_active_pct)]
        if self.is_ensemble:                     # Jacobian-based limits only exist for the ODE solvers
            row += [_NAN, _NAN]
        else:
            row += [float(lambda_max), float(dt_max)]
        self.rows.append(row)

    def add_cells(self, t_h, inc_mask, pi_mask, p38_mask):
        if not self.is_ensemble:
            return
        inc = np.asarray(inc_mask, dtype=bool)
        pi = np.asarray(pi_mask, dtype=bool)
        p38 = np.asarray(p38_mask, dtype=bool)
        new_inc = np.isnan(self.nucleation_h) & inc
        self.nucleation_h[new_inc] = float(t_h)
        new_dead = np.isnan(self.death_h) & (pi | p38)
        self.death_h[new_dead] = float(t_h)
        self.death_route[new_dead] = (pi[new_dead].astype(np.int8) + 2 * p38[new_dead].astype(np.int8))

    # -- analysis ----------------------------------------------------------------------------
    def timeseries(self) -> Dict[str, np.ndarray]:
        arr = np.array(self.rows, dtype=float) if self.rows else np.zeros((0, len(TS_COLUMNS)))
        return {c: arr[:, i] for i, c in enumerate(TS_COLUMNS)}

    def summary(self, status: str = "finished") -> Dict[str, object]:
        ts = self.timeseries()
        t = ts["time_h"]
        n = self.n_cells
        s: Dict[str, object] = {
            "run_id": self.run_id, "saved_at": datetime.now().isoformat(timespec="seconds"),
            "status": status, "scenario": self.scenario, "solver": self.solver, "dt_s": self.dt_s,
            "steps_per_tick": self.steps_per_tick, "n_cells": n if self.is_ensemble else "",
            "seed": self.seed if self.is_ensemble else "",
            "mass_action_dimer": self.mass_action_dimer, "custom_sliders": self.custom_sliders,
            "kp38act": self.params.get("kp38act", _NAN),
            "kproteff_after_24h": self.params.get("kproteff_after_24h", _NAN),
            "kgenROSp38": self.params.get("kgenROSp38", _NAN),
            "kaggMisP": self.params.get("kaggMisP", _NAN),
            "t_end_h": float(t[-1]) if t.size else _NAN, "n_rows": int(t.size),
        }

        def at(col: str, h: float) -> float:
            if not t.size or h < t[0] - 1e-6 or h > t[-1] + 1e-6:
                return _NAN
            return float(np.interp(h, t, ts[col]))

        for h in REPORT_HOURS:
            d = at("death_pct", h)
            inc = at("inclusion_pct", h) if self.is_ensemble else _NAN
            s[f"death_{h}h_pct"] = d
            s[f"inclusion_{h}h_pct"] = inc
            for name, v in (("death", d), ("inclusion", inc)):
                if self.is_ensemble and math.isfinite(v) and n > 0:
                    p = v / 100.0
                    s[f"{name}_{h}h_se_pct"] = 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / n)
                else:
                    s[f"{name}_{h}h_se_pct"] = _NAN
        s["t_seed_h"] = _first_time(t, ts["seq_agg_p"], 1.0, True) if t.size else _NAN
        s["t_proteasome_exhausted_h"] = _first_time(t, ts["free_proteasome"], 1.0, False) if t.size else _NAN
        if self.is_ensemble and n > 0:
            nuc = self.nucleation_h[np.isfinite(self.nucleation_h)]
            s["median_nucleation_h"] = float(np.median(nuc)) if nuc.size else _NAN
            s["cells_with_inclusion_pct"] = 100.0 * nuc.size / n
        else:
            s["median_nucleation_h"] = _NAN
            s["cells_with_inclusion_pct"] = _NAN
        s["final_ros"] = float(ts["ros"][-1]) if t.size else _NAN
        s["final_mrfpu"] = float(ts["mrfpu"][-1]) if t.size else _NAN
        return s

    # -- saving ------------------------------------------------------------------------------
    def save(self, status: str = "finished") -> Dict[str, str]:
        data_dir = os.path.join(self.out_dir, "data")
        fig_dir = os.path.join(self.out_dir, "figures")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        stem = self.run_id if status in ("finished", "unstable") else f"{self.run_id}__partial"
        paths = {"timeseries": os.path.join(data_dir, f"{stem}_timeseries.csv"),
                 "meta": os.path.join(data_dir, f"{stem}_meta.json"),
                 "summary_table": os.path.join(data_dir, SUMMARY_FILE)}

        ts = self.timeseries()
        with open(paths["timeseries"], "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(TS_COLUMNS)
            for row in self.rows:
                w.writerow([_fmt(v) for v in row])

        if self.is_ensemble:
            paths["cells"] = os.path.join(data_dir, f"{stem}_cells.csv")
            route = {0: "", 1: "PI", 2: "p38", 3: "PI+p38"}
            with open(paths["cells"], "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(CELL_COLUMNS)
                for i in range(self.n_cells):
                    w.writerow([i, _fmt(self.nucleation_h[i]), _fmt(self.death_h[i]),
                                route[int(self.death_route[i])]])

        summ = self.summary(status)
        with open(paths["meta"], "w") as fh:
            json.dump(_clean({"summary": summ, "files": {k: os.path.basename(v) for k, v in paths.items()},
                              "notes": NOTES}), fh, indent=2)

        row = {k: (_fmt(v) if not isinstance(v, str) else v) for k, v in summ.items()}
        exists = os.path.exists(paths["summary_table"]) and os.path.getsize(paths["summary_table"]) > 0
        fieldnames = list(SUMMARY_COLUMNS)
        if exists:                                # keep an existing header if columns ever change
            with open(paths["summary_table"], newline="") as fh:
                fieldnames = next(csv.reader(fh))
        with open(paths["summary_table"], "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
            if not exists:
                w.writeheader()
            w.writerow(row)

        try:
            paths["figure"] = os.path.join(fig_dir, f"{stem}.png")
            plot_run(ts, summ, paths["figure"])
        except Exception as exc:                  # a figure problem must never lose the data
            paths.pop("figure", None)
            paths["figure_error"] = f"{type(exc).__name__}: {exc}"
        self.paths = paths
        return paths


def save_run(rec: RunRecorder, status: str = "finished") -> str:
    """Entry point used by NetLogo; returns a one-line message (never raises)."""
    try:
        paths = rec.save(status)
        msg = f"Saved {rec.run_id} ({status}) in {os.path.join(rec.out_dir, 'data')}"
        if "figure_error" in paths:
            msg += f" | figure skipped: {paths['figure_error']}"
        return msg
    except Exception as exc:
        return f"SAVE FAILED: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------------------------
# plotting (matplotlib Figure API only, so it is safe inside the NetLogo/py subprocess)
# ----------------------------------------------------------------------------------------------
_C = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "red": "#D55E00", "pink": "#CC79A7", "black": "#222222", "grey": "#888888"}


def _paper_targets():
    try:
        from .analysis import PAPER_TARGETS
        return PAPER_TARGETS
    except Exception:
        return {}


def plot_run(ts: Dict[str, np.ndarray], summ: Dict[str, object], out_png: str) -> str:
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator

    ens = str(summ["solver"]) == ENSEMBLE_SOLVER
    t = ts["time_h"]
    scen = str(summ["scenario"])
    suffix = " (mean of live cells)" if ens else ""

    fig = Figure(figsize=(12.5, 7.6 if ens else 10.0), layout="constrained")
    gs = fig.add_gridspec(2 if ens else 3, 2, height_ratios=[1, 1] if ens else [1, 1, 0.75])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axE = None if ens else fig.add_subplot(gs[2, :])

    def style(ax, title):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 48)
        ax.xaxis.set_major_locator(MultipleLocator(6))
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)

    def twin(ax, color, label):
        a2 = ax.twinx()
        a2.set_ylabel(label, color=color)
        a2.tick_params(axis="y", colors=color)
        a2.spines["top"].set_visible(False)
        return a2

    # A: oligomer pools and inclusion growth
    style(axA, "A. Oligomers and inclusion growth" + suffix)
    axA.plot(t, ts["polyq_oligomers"], color=_C["orange"], lw=2, label="PolyQ oligomers (AggPolyQ1-5)")
    axA.plot(t, ts["misp_oligomers"], color=_C["red"], lw=2, ls="--", label="MisP oligomers (AggP1-5)")
    axA.set_ylabel("Soluble oligomer copies")
    a2 = twin(axA, _C["blue"], "Inclusion SeqAggP (copies)")
    a2.plot(t, ts["seq_agg_p"], color=_C["blue"], lw=2, label="Inclusion SeqAggP (right axis)")
    h1, l1 = axA.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)

    # B: proteasome and reporter
    style(axB, "B. Proteasome and mRFPu reporter" + suffix)
    axB.plot(t, ts["free_proteasome"], color=_C["green"], lw=2, label="Free proteasome")
    axB.plot(t, ts["bound_proteasome"], color=_C["red"], lw=2, ls="--", label="Bound proteasome")
    axB.set_ylabel("Proteasome copies")
    b2 = twin(axB, _C["blue"], "mRFPu copies")
    b2.plot(t, ts["mrfpu"], color=_C["blue"], lw=2, label="mRFPu (right axis)")
    for ax_, ys in ((axB, ("free_proteasome", "bound_proteasome")), (b2, ("mrfpu",))):
        top = max([float(np.nanmax(ts[y])) for y in ys if ts[y].size] + [1.0])
        ax_.set_ylim(0, 1.45 * top)           # head-room so the legend never covers a curve
    ex = summ.get("t_proteasome_exhausted_h")
    if isinstance(ex, float) and math.isfinite(ex):
        axB.axvline(ex, color=_C["grey"], ls=":", lw=1.4, label=f"Free proteasome < 1 at {ex:.2f} h")
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=8, frameon=False)

    # C: ROS and p38
    style(axC, "C. ROS and p38 MAPK activation" + suffix)
    axC.plot(t, ts["ros"], color=_C["pink"], lw=2, label="ROS copies")
    axC.set_ylabel("ROS copies", color=_C["pink"])
    c2 = twin(axC, _C["black"], "Active p38_P (%)")
    c2.plot(t, ts["p38_active_pct"], color=_C["black"], lw=2, ls="--", label="p38_P (% of total, right axis)")
    h1, l1 = axC.get_legend_handles_labels(); h2, l2 = c2.get_legend_handles_labels()
    axC.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)

    # D: death (and inclusion frequency for ensembles) with paper values
    style(axD, "D. Cumulative death" + (" and inclusion frequency" if ens else ""))
    axD.plot(t, ts["death_pct"], color=_C["black"], lw=2.4, label="Total death")
    axD.plot(t, ts["death_pi_route_pct"], color=_C["pink"], lw=1.6, ls="--", label="via proteasome (PI) route")
    axD.plot(t, ts["death_p38_route_pct"], color=_C["sky"], lw=1.6, ls="-.", label="via p38 route")
    if ens:
        axD.plot(t, ts["inclusion_pct"], color=_C["orange"], lw=2.2, label="Cells with inclusion")
    else:
        sd = summ.get("t_seed_h")
        if isinstance(sd, float) and math.isfinite(sd):
            axD.axvline(sd, color=_C["orange"], ls="--", lw=1.4, label=f"Inclusion seed, all cells ({sd:.2f} h)")
    if "pi" in scen:
        axD.axvline(24, color=_C["grey"], ls=":", lw=1.2)
    pt = _paper_targets()
    drawn = False
    d30 = pt.get("fig5A_death_30h_pct", {}).get(scen, pt.get("fig7A_death_30h_pct", {}).get(scen))
    if d30 is not None:
        axD.plot([30], [d30], "x", color=_C["black"], ms=9, mew=2.2, label="Paper (read by eye)")
        drawn = True
    if ens:
        inc = pt.get("fig5B_inclusion_freq_pct", {})
        xs = [h for h in sorted(inc) if scen in inc[h]]
        if xs:
            axD.plot(xs, [inc[h][scen] for h in xs], "x", color=_C["orange"], ms=9, mew=2.2,
                     label=None if drawn else "Paper (read by eye)")
    axD.set_ylabel("% of cells")
    axD.set_ylim(bottom=0)
    axD.legend(loc="upper left", fontsize=8, frameon=False)
    for ax in (axC, axD):
        ax.set_xlabel("Time (h)")

    # E: stability ceiling (ODE only)
    if axE is not None:
        style(axE, "E. Explicit-solver stability: dt must stay below dt_max = 2 / |lambda_max|")
        axE.plot(t, ts["dt_max_s"], color=_C["red"], lw=2, label="dt_max (Jacobian limit)")
        axE.axhline(float(summ["dt_s"]), color=_C["blue"], lw=2, ls="--", label=f"step size used: {float(summ['dt_s']):g} s")
        axE.set_yscale("log")
        axE.set_ylabel("seconds (log scale)")
        axE.set_xlabel("Time (h)")
        axE.legend(loc="upper right", fontsize=8, frameon=False)
    for ax in (axA, axB):
        ax.set_xlabel("Time (h)")

    # titles
    bits = [scen, str(summ["solver"]), f"dt = {float(summ['dt_s']):g} s"]
    if ens:
        bits.append(f"N = {summ['n_cells']} cells, seed {summ['seed']}")
    if summ.get("custom_sliders") in (True, "True"):
        bits.append("custom sliders ON")
    if str(summ.get("status")) != "finished":
        bits.append(f"status: {summ['status']}")

    def g(k):
        v = summ.get(k)
        return f"{v:.2f}" if isinstance(v, float) and math.isfinite(v) else "n/a"
    res = f"death at 30 h: {g('death_30h_pct')} %   |   death at 48 h: {g('death_48h_pct')} %"
    if ens:
        res += f"   |   inclusions at 48 h: {g('inclusion_48h_pct')} %   |   median inclusion time: {g('median_nucleation_h')} h"
    else:
        res += f"   |   inclusion seed: {g('t_seed_h')} h   |   free proteasome exhausted: {g('t_proteasome_exhausted_h')} h"
    fig.suptitle("  |  ".join(bits) + "\n" + res, fontsize=11)
    fig.savefig(out_png, dpi=300)
    return out_png


# ----------------------------------------------------------------------------------------------
# command line: re-draw figures from saved data
# ----------------------------------------------------------------------------------------------
def _read_timeseries(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = {h: [] for h in header}
        for row in rd:
            for h, v in zip(header, row):
                cols[h].append(_f(v))
    return {h: np.array(v, dtype=float) for h, v in cols.items()}


def replot(out_dir: str = DEFAULT_OUT_DIR) -> int:
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if not name.endswith("_meta.json"):
            continue
        with open(os.path.join(data_dir, name)) as fh:
            meta = json.load(fh)
        summ = {k: (_NAN if v is None else v) for k, v in meta["summary"].items()}
        ts = _read_timeseries(os.path.join(data_dir, meta["files"]["timeseries"]))
        stem = name[: -len("_meta.json")]
        plot_run(ts, summ, os.path.join(fig_dir, f"{stem}.png"))
        n += 1
        print(f"redrew {stem}.png")
    print(f"{n} figure(s) written to {fig_dir}")
    return n


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replot":
        replot(argv[1] if len(argv) > 1 else DEFAULT_OUT_DIR)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
