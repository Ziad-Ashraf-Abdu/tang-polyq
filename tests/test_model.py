# File: tang-polyq/tests/test_model.py
import unittest
import numpy as np
from tang_polyq.model import load_default_model
from tang_polyq.scenarios import get_scenarios
from tang_polyq.solvers import solve_radau
from tang_polyq.analysis import compute_metrics

class TestTangPolyQModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_default_model(use_mass_action_dimer=True)
        cls.scenarios = get_scenarios()

    def test_network_dimensions_and_rank(self):
        self.assertEqual(len(self.model.species_ids), 25)
        self.assertEqual(len(self.model.boundary_ids), 2)
        self.assertEqual(len(self.model.reaction_ids), 72)
        self.assertEqual(len(self.model.parameters), 47)
        rank = np.linalg.matrix_rank(self.model.stoich_matrix)
        self.assertEqual(rank, 24)

    def test_p38_conservation_law_and_radau_reference(self):
        res = solve_radau(self.model, self.scenarios["untreated"])
        self.assertTrue(res.success)
        p38 = res.get_species(self.model, "p38")
        p38_p = res.get_species(self.model, "p38_P")
        
        # The total p38 pool in BIOMD0000000285 is 100 copies
        np.testing.assert_allclose(p38 + p38_p, 100.0, rtol=1e-7, atol=1e-7)

        met = compute_metrics(self.model, res)
        self.assertAlmostEqual(met.death_at_30h_pct, 14.23, places=1)
        self.assertAlmostEqual(met.death_at_48h_pct, 34.76, places=1)
        self.assertAlmostEqual(met.t_inclusion_seed_h, 6.70, places=1)
        self.assertAlmostEqual(met.t_proteasome_exhaust_h, 19.77, places=1)

if __name__ == "__main__":
    unittest.main()