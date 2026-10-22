# File: tang-polyq/src/tang_polyq/scenarios.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from .model import SBMLModel

@dataclass
class Scenario:
    name: str
    label: str
    param_overrides: Dict[str, float] = field(default_factory=dict)
    y0_overrides: Dict[str, float] = field(default_factory=dict)
    switch_time: Optional[float] = None
    post_switch_overrides: Dict[str, float] = field(default_factory=dict)

    def get_initial_state(self, model: SBMLModel) -> np.ndarray:
        y0 = model.y0.copy()
        for sp, val in self.y0_overrides.items():
            y0[model.species_map[sp]] = val
        return y0

    def get_params_at(self, model: SBMLModel, t: float) -> Dict[str, float]:
        p = model.parameters.copy()
        p.update(self.param_overrides)
        if self.switch_time is not None and t >= self.switch_time:
            p.update(self.post_switch_overrides)
        return p

def get_scenarios() -> Dict[str, Scenario]:
    return {
        "untreated": Scenario(
            name="untreated",
            label="Untreated (PolyQ)",
        ),
        "p38_inhibitor": Scenario(
            name="p38_inhibitor",
            label="p38 Inhibitor (kp38act = 0.05, Text S1)",
            param_overrides={"kp38act": 0.05},
        ),
        "pi_24h": Scenario(
            name="pi_24h",
            label="Proteasome Inhibition at 24h (kproteff = 0.05)",
            switch_time=24.0 * 3600.0,
            post_switch_overrides={"kproteff": 0.05},
        ),
        "pi_24h_p38_inhibitor": Scenario(
            name="pi_24h_p38_inhibitor",
            label="PI at 24h + p38 Inhibitor (kp38act = 0.05)",
            param_overrides={"kp38act": 0.05},
            switch_time=24.0 * 3600.0,
            post_switch_overrides={"kproteff": 0.05},
        ),
        "no_p38_ros_feedback": Scenario(
            name="no_p38_ros_feedback",
            label="No p38->ROS Feedback (kgenROSp38 = 0, Fig 7A Untreated)",
            param_overrides={"kgenROSp38": 0.0},
        ),
        "pi_24h_no_feedback": Scenario(
            name="pi_24h_no_feedback",
            label="PI at 24h + No p38->ROS Feedback (Fig 7A PI)",
            param_overrides={"kgenROSp38": 0.0},
            switch_time=24.0 * 3600.0,
            post_switch_overrides={"kproteff": 0.05},
        ),
        "p38_inh_no_feedback": Scenario(
            name="p38_inh_no_feedback",
            label="p38 Inhibitor + No p38->ROS Feedback (Fig 7A p38inh)",
            param_overrides={"kgenROSp38": 0.0, "kp38act": 0.05},
        ),
        "no_basal_ros_gen": Scenario(
            name="no_basal_ros_gen",
            label="No Basal ROS Generation (kgenROS = 0)",
            param_overrides={"kgenROS": 0.0},
        ),
        "no_polyq_control": Scenario(
            name="no_polyq_control",
            label="No-PolyQ Control (PolyQ = 0)",
            y0_overrides={"PolyQ": 0.0},
            param_overrides={"ksynPolyQ": 0.0},
        ),
        "no_polyq_no_agg": Scenario(
            name="no_polyq_no_agg",
            label="No-PolyQ Control + No MisP Agg (kaggMisP = 0)",
            y0_overrides={"PolyQ": 0.0},
            param_overrides={"ksynPolyQ": 0.0, "kaggMisP": 0.0},
        ),
    }