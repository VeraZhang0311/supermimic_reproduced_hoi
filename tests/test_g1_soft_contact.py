"""Real MuJoCo contact probes, independent of Isaac Lab and GPU availability."""
from pathlib import Path
import unittest

from supermimic.g1_physics import ASSET_DIR


@unittest.skipUnless((ASSET_DIR / "manifest.json").is_file(), "Prepare verified hw3 hand assets before real contact probes")
class SoftContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from supermimic.g1_validation import soft_hand_probes
        cls.report = soft_hand_probes()

    def test_actual_finger_hulls_meet_incremental_press_target(self):
        # The force balance and measured displacement come from actual dynamics.
        for curve in self.report["curves"]:
            if curve["part"] != "fingers":
                continue
            self.assertGreaterEqual(curve["incremental_N_mm"], .5)
            self.assertLessEqual(curve["incremental_N_mm"], .6)
            p1, p5, p10 = curve["presses"]
            self.assertLess(p1["penetration_mm"], p5["penetration_mm"])
            self.assertLess(p5["penetration_mm"], p10["penetration_mm"])
            for p in curve["presses"]:
                self.assertAlmostEqual(p["normal_force_N"], p["load_N"], delta=.03)
                self.assertLess(p["settled_velocity_max_m_s"], .001)

    def test_thumb_curves_are_reported_without_false_equal_stiffness_claim(self):
        # Uniform material + different convex geometry does not imply identical
        # pad stiffness. In particular the right thumb can have fewer contacts.
        thumbs = [c for c in self.report["curves"] if c["part"] == "thumb"]
        self.assertEqual({c["side"] for c in thumbs}, {"left", "right"})
        for curve in thumbs:
            self.assertGreater(curve["incremental_N_mm"], 0)
            self.assertTrue(all(p["penetration_mm"] > 0 for p in curve["presses"]))
            for p in curve["presses"]:
                self.assertAlmostEqual(p["normal_force_N"], p["load_N"], delta=.03)
                self.assertLess(p["settled_velocity_max_m_s"], .001)
        self.assertFalse(self.report["physx_equivalence"])

    def test_mass_range_settles_and_unloading_removes_force(self):
        probes = self.report["mass_release_probes"]
        self.assertEqual([p["mass_kg"] for p in probes], [.25, 1., 4.])
        for p in probes:
            self.assertAlmostEqual(p["normal_force_N"], 5., delta=.03)
            self.assertLess(p["settled_velocity_max_m_s"], .001)
            self.assertLess(p["release"]["normal_force_N"], .01)
            self.assertLess(p["release"]["penetration_mm"], .01)
        # Nominal inverse-weight normalization should survive this isolated
        # mass change. This is not an articulated effective-mass assertion.
        compression = [p["penetration_mm"] for p in probes]
        self.assertLess(max(compression)-min(compression), .05)

    def test_rigid_palm_and_same_geometry_control_stay_hard(self):
        soft = self.report["mass_release_probes"][1]["penetration_mm"]
        for key in ("rigid_same_geometry_control", "actual_palm_probe"):
            p = self.report[key]
            self.assertLess(p["penetration_mm"], soft/5)
            self.assertAlmostEqual(p["normal_force_N"], 5., delta=.03)

    def test_hand_surface_friction_below_and_above_coulomb_limit(self):
        low, high = self.report["surface_friction"]["results"]
        self.assertLess(abs(low["slide_velocity_m_s"]), .001)
        self.assertGreater(high["slide_velocity_m_s"], .1)
        self.assertGreater(high["slide_distance_m"], 5*low["slide_distance_m"])

    def test_selected_solver_reduces_baseline_creep_without_stiffening_normal_contact(self):
        low = self.report["surface_friction"]["results"][0]
        baseline = self.report["pyramidal_friction_baseline"]["results"][0]
        self.assertGreater(abs(baseline["slide_velocity_m_s"]), .001)
        self.assertLess(abs(low["slide_velocity_m_s"]), abs(baseline["slide_velocity_m_s"])/10)
        # The finger press test separately holds the original response target.
        self.assertTrue(all(.5 <= c["incremental_N_mm"] <= .6
                            for c in self.report["curves"] if c["part"] == "fingers"))

    def test_drop_convergence_reports_out_of_calibration_high_load(self):
        drops = self.report["drop_sensitivity"]
        self.assertEqual([p["mass_kg"] for p in drops], [.25, 1., 4.])
        for p in drops:
            self.assertTrue(p["finite"])
            self.assertLess(p["settled_velocity_max_m_s"], .001)
            self.assertAlmostEqual(p["final_normal_force_N"], p["nominal_load_N"], delta=.05)
        self.assertGreater(drops[-1]["nominal_load_N"], 10.)
        self.assertFalse(drops[-1]["within_source_calibration_force_range"])

    def test_production_report_enforces_response_criteria(self):
        from supermimic.g1_validation import check_soft_hand_probes
        check_soft_hand_probes(self.report)


if __name__ == "__main__":
    unittest.main()
