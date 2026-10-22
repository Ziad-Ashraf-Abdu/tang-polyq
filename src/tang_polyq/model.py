# File: tang-polyq/src/tang_polyq/model.py
import os
import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable

SBML_NS = {
    "sbml": "http://www.sbml.org/sbml/level2/version4",
    "math": "http://www.w3.org/1998/Math/MathML",
}

DIMER_SPECIES = [
    "PolyQ", "MisP",
    "AggPolyQ1", "AggPolyQ2", "AggPolyQ3", "AggPolyQ4", "AggPolyQ5",
    "AggP1", "AggP2", "AggP3", "AggP4", "AggP5", "SeqAggP",
]

def _parse_mathml_node(node: ET.Element) -> str:
    tag = node.tag.split("}")[-1]
    if tag == "math":
        return _parse_mathml_node(list(node)[0])
    elif tag == "ci":
        return node.text.strip()
    elif tag == "cn":
        return str(float(node.text.strip()))
    elif tag == "apply":
        children = list(node)
        op = children[0].tag.split("}")[-1]
        args = [_parse_mathml_node(c) for c in children[1:]]
        if op == "times":
            return "(" + " * ".join(args) + ")"
        elif op == "plus":
            return "(" + " + ".join(args) + ")"
        elif op == "minus":
            if len(args) == 1:
                return f"(-{args[0]})"
            return f"({args[0]} - {args[1]})"
        elif op == "divide":
            return f"({args[0]} / {args[1]})"
        elif op == "power":
            return f"({args[0]} ** {args[1]})"
        else:
            raise ValueError(f"Unsupported MathML op: {op}")
    else:
        raise ValueError(f"Unsupported MathML tag: {tag}")

@dataclass
class SBMLModel:
    species_ids: List[str]
    boundary_ids: List[str]
    y0: np.ndarray
    boundary_values: Dict[str, float]
    parameters: Dict[str, float]
    reaction_ids: List[str]
    stoich_matrix: np.ndarray          # Shape: (25, 72)
    reactant_matrix: np.ndarray        # Shape: (25, 72)
    rate_exprs: List[str]
    use_mass_action_dimer: bool = True
    _rate_func: Callable = field(repr=False, default=None)
    _propensity_func: Callable = field(repr=False, default=None)

    def __post_init__(self):
        self.species_map = {s: i for i, s in enumerate(self.species_ids)}
        self.reaction_map = {r: i for i, r in enumerate(self.reaction_ids)}
        self._compile_rates(stochastic=False)
        self._compile_rates(stochastic=True)

    def _compile_rates(self, stochastic: bool = False):
        lines = ["def _rates(y, p):"]
        for i, s in enumerate(self.species_ids):
            if stochastic:
                lines.append(f"    {s} = np.maximum(y[{i}], 0.0)")
            else:
                lines.append(f"    {s} = y[{i}]")
        for b, val in self.boundary_values.items():
            lines.append(f"    {b} = {val}")
        for k in self.parameters.keys():
            lines.append(f"    {k} = p['{k}']")

        lines.append(f"    v = np.empty(({len(self.reaction_ids)},) + np.shape(y[0]), dtype=np.float64)")
        for j, expr in enumerate(self.rate_exprs):
            if stochastic:
                lines.append(f"    v[{j}] = np.maximum({expr}, 0.0)")
            else:
                det_expr = expr
                if self.use_mass_action_dimer:
                    # Documented switch: replaces discrete stochastic X*(X-1)*0.5 with continuous X*X*0.5
                    # so fractional concentrations (0 < X < 1) in ODEs do not produce negative dimerization rates.
                    for s in DIMER_SPECIES:
                        det_expr = det_expr.replace(f"({s} - 1.0)", s)
                lines.append(f"    v[{j}] = {det_expr}")
        lines.append("    return v")
        code = "\n".join(lines)
        ns = {"np": np}
        exec(code, ns)
        if stochastic:
            self._propensity_func = ns["_rates"]
        else:
            self._rate_func = ns["_rates"]

    def rates(self, t: float, y: np.ndarray, params: Dict[str, float] = None) -> np.ndarray:
        p = self.parameters if params is None else params
        return self._rate_func(y, p)

    def propensities(self, t: float, y: np.ndarray, params: Dict[str, float] = None) -> np.ndarray:
        p = self.parameters if params is None else params
        return self._propensity_func(y, p)

    def rhs(self, t: float, y: np.ndarray, params: Dict[str, float] = None) -> np.ndarray:
        v = self.rates(t, y, params)
        return self.stoich_matrix @ v

    def summary(self) -> str:
        rank = np.linalg.matrix_rank(self.stoich_matrix)
        return (
            f"Tang et al. (2010) SBML Model (BIOMD0000000285)\n"
            f"  Dynamic Species  : {len(self.species_ids)}\n"
            f"  Boundary Species : {len(self.boundary_ids)} ({', '.join(self.boundary_ids)})\n"
            f"  Reactions        : {len(self.reaction_ids)}\n"
            f"  Kinetic Params   : {len(self.parameters)}\n"
            f"  Mass-Action Dimer: {self.use_mass_action_dimer} (X*(X-1) -> X^2 in ODE)\n"
            f"  Stoich Rank      : {rank} / {len(self.species_ids)} "
            f"({len(self.species_ids) - rank} conservation law: p38 + p38_P = const)"
        )

def load_default_model(xml_path: str = None, use_mass_action_dimer: bool = True) -> SBMLModel:
    if xml_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        xml_path = os.path.join(base_dir, "data", "BIOMD0000000285_url.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"SBML file not found at {xml_path}.")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    model_el = root.find("sbml:model", SBML_NS)

    species_ids = []
    boundary_ids = []
    y0_list = []
    boundary_values = {}

    for sp in model_el.find("sbml:listOfSpecies", SBML_NS).findall("sbml:species", SBML_NS):
        sid = sp.attrib["id"]
        val = float(sp.attrib.get("initialAmount", sp.attrib.get("initialConcentration", 0.0)))
        if sp.attrib.get("boundaryCondition", "false").lower() == "true":
            boundary_ids.append(sid)
            boundary_values[sid] = val
        else:
            species_ids.append(sid)
            y0_list.append(val)

    parameters = {}
    for par in model_el.find("sbml:listOfParameters", SBML_NS).findall("sbml:parameter", SBML_NS):
        pid = par.attrib["id"]
        # Skip 'oligomers' as it is an SBML assignment rule, not a kinetic parameter
        if pid == "oligomers":
            continue
        parameters[pid] = float(par.attrib.get("value", 0.0))

    reactions = model_el.find("sbml:listOfReactions", SBML_NS).findall("sbml:reaction", SBML_NS)
    n_sp = len(species_ids)
    n_rxn = len(reactions)
    stoich_matrix = np.zeros((n_sp, n_rxn), dtype=np.float64)
    reactant_matrix = np.zeros((n_sp, n_rxn), dtype=np.float64)
    reaction_ids = []
    rate_exprs = []
    sp_idx = {s: i for i, s in enumerate(species_ids)}

    for j, rxn in enumerate(reactions):
        rid = rxn.attrib["id"]
        reaction_ids.append(rid)
        reactants = rxn.find("sbml:listOfReactants", SBML_NS)
        if reactants is not None:
            for sr in reactants.findall("sbml:speciesReference", SBML_NS):
                sname = sr.attrib["species"]
                st = float(sr.attrib.get("stoichiometry", 1.0))
                if sname in sp_idx:
                    stoich_matrix[sp_idx[sname], j] -= st
                    reactant_matrix[sp_idx[sname], j] += st
        products = rxn.find("sbml:listOfProducts", SBML_NS)
        if products is not None:
            for sr in products.findall("sbml:speciesReference", SBML_NS):
                sname = sr.attrib["species"]
                st = float(sr.attrib.get("stoichiometry", 1.0))
                if sname in sp_idx:
                    stoich_matrix[sp_idx[sname], j] += st

        kl = rxn.find("sbml:kineticLaw", SBML_NS)
        math_el = kl.find("math:math", SBML_NS)
        rate_exprs.append(_parse_mathml_node(math_el))

    return SBMLModel(
        species_ids=species_ids,
        boundary_ids=boundary_ids,
        y0=np.array(y0_list, dtype=np.float64),
        boundary_values=boundary_values,
        parameters=parameters,
        reaction_ids=reaction_ids,
        stoich_matrix=stoich_matrix,
        reactant_matrix=reactant_matrix,
        rate_exprs=rate_exprs,
        use_mass_action_dimer=use_mass_action_dimer,
    )