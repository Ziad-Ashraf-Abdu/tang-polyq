# File: tang-polyq/tests/test_netlogo_export.py
import csv
import json
import os
import tempfile
import unittest

import numpy as np

from tang_polyq.model import load_default_model
from tang_polyq.netlogo_export import (
    ENSEMBLE_SOLVER, SUMMARY_COLUMNS, TS_COLUMNS, RunRecorder, effective_params, save_run,
)
from tang_polyq.scenarios import get_scenarios


def _recorder(out_dir, solver="RK2 (ODE)", n_cells=4):
    return RunRecorder("untreated", solver, 2.0, 60, n_cells, 42, True, False,
                       params={"kp38act": 1.0, "kproteff_after_24h": 1.0, "kgenROSp38": 7e-4, "kaggMisP": 1e-11},
                       out_dir=out_dir)


def _feed_linear(rec, hours=(0, 12, 24, 30, 36, 42, 48)):
    """death grows 1 % per hour, SeqAggP reaches 1 at 12 h, free proteasome falls below 1 at 36 h"""
    for h in hours:
        rec.add(h, 1.0 * h, 0.0, 1.0 * h, 0.0 if h < 12 else 100.0, 100.0, 200.0, 0.0 if h < 12 else 5.0,
                0.5 if h >= 36 else 500.0, 1.0, 20.0, 300.0, 10.0, 0.01, 200.0)


class TestNetLogoExport(unittest.TestCase):
    def test_ode_run_is_saved_with_expected_files_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _recorder(d)
            _feed_linear(rec)
            msg = save_run(rec, "finished")
            self.assertTrue(msg.startswith("Saved"), msg)
            data = os.path.join(d, "data")
            files = os.listdir(data)
            self.assertTrue(any(f.endswith("_timeseries.csv") for f in files))
            self.assertTrue(any(f.endswith("_meta.json") for f in files))
            self.assertFalse(any(f.endswith("_cells.csv") for f in files))  # ODE: no per-cell file
            self.assertTrue(os.path.exists(os.path.join(data, "netlogo_run_summary.csv")))

            with open([os.path.join(data, f) for f in files if f.endswith("_timeseries.csv")][0]) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(tuple(rows[0].keys()), TS_COLUMNS)
            self.assertEqual(len(rows), 7)

            with open(os.path.join(data, "netlogo_run_summary.csv")) as fh:
                summ = list(csv.DictReader(fh))[0]
            self.assertEqual(list(summ.keys()), list(SUMMARY_COLUMNS))
            self.assertAlmostEqual(float(summ["death_30h_pct"]), 30.0)
            self.assertAlmostEqual(float(summ["death_48h_pct"]), 48.0)
            self.assertAlmostEqual(float(summ["t_seed_h"]), 12.0)
            self.assertAlmostEqual(float(summ["t_proteasome_exhausted_h"]), 36.0)
            self.assertEqual(summ["status"], "finished")

    def test_partial_save_does_not_overwrite_final_and_hours_outside_run_are_nan(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _recorder(d)
            _feed_linear(rec, hours=(0, 12, 24))
            save_run(rec, "partial")
            names = os.listdir(os.path.join(d, "data"))
            self.assertTrue(any("__partial_timeseries.csv" in n for n in names))
            with open(os.path.join(d, "data", "netlogo_run_summary.csv")) as fh:
                summ = list(csv.DictReader(fh))[0]
            self.assertEqual(summ["death_30h_pct"], "nan")
            self.assertAlmostEqual(float(summ["death_24h_pct"]), 24.0)

    def test_ensemble_per_cell_times_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _recorder(d, solver=ENSEMBLE_SOLVER, n_cells=4)
            rec.add_cells(10.0, [True, False, False, False], [False, False, False, False], [False, False, False, False])
            rec.add_cells(20.0, [True, True, False, False], [False, True, False, False], [False, False, False, False])
            rec.add_cells(30.0, [True, True, True, False], [False, True, False, False], [False, False, True, False])
            np.testing.assert_allclose(rec.nucleation_h, [10, 20, 30, np.nan])
            np.testing.assert_allclose(rec.death_h, [np.nan, 20, 30, np.nan])
            for h in (0, 24, 30, 36, 42, 48):
                rec.add(h, 25.0, 10.0, 15.0, 75.0, 1, 1, 1, 1, 1, 1, 1, 1)
            s = rec.summary("finished")
            self.assertAlmostEqual(s["median_nucleation_h"], 20.0)
            self.assertAlmostEqual(s["cells_with_inclusion_pct"], 75.0)
            self.assertAlmostEqual(s["death_30h_se_pct"], 100 * np.sqrt(0.25 * 0.75 / 4))
            save_run(rec, "finished")
            names = os.listdir(os.path.join(d, "data"))
            cells_file = [n for n in names if n.endswith("_cells.csv")][0]
            with open(os.path.join(d, "data", cells_file)) as fh:
                cells = list(csv.DictReader(fh))
            self.assertEqual([c["death_route"] for c in cells], ["", "PI", "p38", ""])

    def test_meta_json_is_valid_json_even_with_nan(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _recorder(d)
            _feed_linear(rec, hours=(0, 12))
            save_run(rec, "partial")
            meta = [f for f in os.listdir(os.path.join(d, "data")) if f.endswith("_meta.json")][0]
            with open(os.path.join(d, "data", meta)) as fh:
                txt = fh.read()
            self.assertNotIn("NaN", txt)
            self.assertEqual(json.loads(txt)["summary"]["scenario"], "untreated")

    def test_effective_params_follow_scenario_unless_custom(self):
        model = load_default_model(use_mass_action_dimer=True)
        scen = get_scenarios()["pi_24h_p38_inhibitor"]
        p0, p1 = scen.get_params_at(model, 0.0), scen.get_params_at(model, 1e12)
        eff = effective_params(p0, p1, False, {})
        self.assertEqual(eff["kp38act"], 0.05)
        self.assertEqual(eff["kproteff_after_24h"], 0.05)
        custom = effective_params(p0, p1, True, {"kp38act": 0.5, "kaggMisP": 0.0})
        self.assertEqual(custom["kp38act"], 0.5)
        self.assertEqual(custom["kaggMisP"], 0.0)
        self.assertEqual(custom["kproteff_after_24h"], 0.05)

    def test_save_never_raises(self):
        rec = _recorder("/proc/does/not/exist")
        _feed_linear(rec, hours=(0, 12))
        self.assertTrue(save_run(rec, "partial").startswith("SAVE FAILED"))


if __name__ == "__main__":
    unittest.main()
