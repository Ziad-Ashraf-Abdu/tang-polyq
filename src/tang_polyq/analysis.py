# File: tang-polyq/src/tang_polyq/analysis.py
import numpy as np
from dataclasses import dataclass
from .model import SBMLModel
from .solvers import SimulationResult

# Digitized reference values from Tang et al. (2010) PLoS Comput Biol 6(9): e1000944
PAPER_TARGETS = {
    "fig5A_death_30h_pct": {
        "untreated": 21.0,      # Fig 5A: PolyQ untreated (~21%)
        "pi_24h": 18.0,         # Fig 5A: PolyQ + PI (~18%)
        "p38_inhibitor": 8.0,   # Fig 5A: PolyQ + p38inh (~8%)
    },
    "fig7A_death_30h_pct": {
        "no_p38_ros_feedback": 6.7,      # Fig 7A: PolyQ without p38->ROS feedback (~6.7%)
        "pi_24h_no_feedback": 9.8,       # Fig 7A: PolyQ + PI without feedback (~9.8%)
        "p38_inh_no_feedback": 4.7,      # Fig 7A: PolyQ + p38inh without feedback (~4.7%)
    },
    "fig5B_inclusion_freq_pct": {
        24: {"untreated": 13.0, "pi_24h": 13.0},
        30: {"untreated": 20.0, "pi_24h": 26.0},
        36: {"untreated": 36.0, "pi_24h": 51.0},
        42: {"untreated": 48.0, "pi_24h": 76.0},
        48: {"untreated": 60.0, "pi_24h": 77.0},
    },
}

@dataclass
class DeterministicMetrics:
    scenario_name: str
    t_hours: np.ndarray
    survival_pct: np.ndarray
    death_pct: np.ndarray
    death_PI_only_pct: np.ndarray
    death_p38_only_pct: np.ndarray
    polyq_oligomers: np.ndarray        # Sum of AggPolyQ1..5 (~126 at 30h)
    misp_oligomers: np.ndarray         # Sum of AggP1..5 (~2,148 at 30h)
    total_oligomers: np.ndarray        # SBML 'oligomers' = AggPolyQ1..5 + AggP1..5
    inclusion_size: np.ndarray         # SeqAggP
    free_proteasome: np.ndarray
    sequestered_proteasome: np.ndarray
    mRFPu: np.ndarray
    ros: np.ndarray
    p38_p_frac: np.ndarray
    t_inclusion_seed_h: float
    t_proteasome_exhaust_h: float
    death_at_30h_pct: float
    death_at_48h_pct: float

def compute_metrics(model: SBMLModel, res: SimulationResult) -> DeterministicMetrics:
    def safe_get(name: str) -> np.ndarray:
        return res.get_species(model, name) if name in model.species_map else np.zeros_like(res.t_hours)

    pi_haz = safe_get("PIdeath")
    p38_haz = safe_get("p38death")
    total_haz = pi_haz + p38_haz

    surv = np.exp(-np.maximum(total_haz, 0.0)) * 100.0
    death = 100.0 - surv
    death_pi = (1.0 - np.exp(-np.maximum(pi_haz, 0.0))) * 100.0
    death_p38 = (1.0 - np.exp(-np.maximum(p38_haz, 0.0))) * 100.0

    polyq_olig = sum(safe_get(f"AggPolyQ{i}") for i in range(1, 6))
    misp_olig = sum(safe_get(f"AggP{i}") for i in range(1, 6))
    total_olig = polyq_olig + misp_olig

    seq = safe_get("SeqAggP")
    prot_free = safe_get("Proteasome")
    prot_seq = (
        safe_get("AggP_Proteasome")
        + safe_get("PolyQ_Proteasome")
        + safe_get("MisP_Proteasome")
        + safe_get("mRFPu_Proteasome")
    )
    mrfpu = safe_get("mRFPu")
    ros = safe_get("ROS")
    p38 = safe_get("p38")
    p38_p = safe_get("p38_P")
    p38_frac = p38_p / np.maximum(p38 + p38_p, 1e-12)

    seed_indices = np.where(seq >= 1.0)[0]
    t_seed = float(res.t_hours[seed_indices[0]]) if len(seed_indices) > 0 else np.nan

    exh_indices = np.where(prot_free < 1.0)[0]
    t_exh = float(res.t_hours[exh_indices[0]]) if len(exh_indices) > 0 else np.nan

    idx30 = int(np.argmin(np.abs(res.t_hours - 30.0)))
    idx48 = int(np.argmin(np.abs(res.t_hours - 48.0)))

    return DeterministicMetrics(
        scenario_name=res.scenario_name,
        t_hours=res.t_hours,
        survival_pct=surv,
        death_pct=death,
        death_PI_only_pct=death_pi,
        death_p38_only_pct=death_p38,
        polyq_oligomers=polyq_olig,
        misp_oligomers=misp_olig,
        total_oligomers=total_olig,
        inclusion_size=seq,
        free_proteasome=prot_free,
        sequestered_proteasome=prot_seq,
        mRFPu=mrfpu,
        ros=ros,
        p38_p_frac=p38_frac,
        t_inclusion_seed_h=t_seed,
        t_proteasome_exhaust_h=t_exh,
        death_at_30h_pct=float(death[idx30]),
        death_at_48h_pct=float(death[idx48]),
    )