from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.tensor_metrics import tensor_metrics_from_velocity_gradients


class TensorMetricsTests(unittest.TestCase):
    def test_rigid_rotation_is_rotation_dominated(self) -> None:
        angular_rate = 2.0
        gradient = np.array([[[0.0, -angular_rate, 0.0], [angular_rate, 0.0, 0.0], [0.0, 0.0, 0.0]]])

        metrics = tensor_metrics_from_velocity_gradients(gradient)

        self.assertAlmostEqual(metrics["S_norm2"][0], 0.0)
        self.assertAlmostEqual(metrics["Omega_norm2"][0], 2.0 * angular_rate**2)
        self.assertAlmostEqual(metrics["Q"][0], angular_rate**2)
        self.assertAlmostEqual(metrics["lambda2"][0], -angular_rate**2)
        self.assertAlmostEqual(metrics["omega_normalized"][0], 1.0)
        self.assertFalse(metrics["zero_tensor"][0])

    def test_pure_strain_is_strain_dominated(self) -> None:
        strain_rate = 3.0
        gradient = np.array([[[strain_rate, 0.0, 0.0], [0.0, -strain_rate, 0.0], [0.0, 0.0, 0.0]]])

        metrics = tensor_metrics_from_velocity_gradients(gradient)

        self.assertAlmostEqual(metrics["S_norm2"][0], 2.0 * strain_rate**2)
        self.assertAlmostEqual(metrics["Omega_norm2"][0], 0.0)
        self.assertAlmostEqual(metrics["Q"][0], -strain_rate**2)
        self.assertAlmostEqual(metrics["lambda2"][0], strain_rate**2)
        self.assertAlmostEqual(metrics["omega_normalized"][0], 0.0)

    def test_simple_shear_is_q_and_omega_neutral(self) -> None:
        shear_rate = 4.0
        gradient = np.array([[[0.0, shear_rate, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

        metrics = tensor_metrics_from_velocity_gradients(gradient)

        self.assertAlmostEqual(metrics["S_norm2"][0], 0.5 * shear_rate**2)
        self.assertAlmostEqual(metrics["Omega_norm2"][0], 0.5 * shear_rate**2)
        self.assertAlmostEqual(metrics["Q"][0], 0.0)
        self.assertAlmostEqual(metrics["lambda2"][0], 0.0)
        self.assertAlmostEqual(metrics["omega_normalized"][0], 0.5)
        self.assertFalse(metrics["zero_tensor"][0])

    def test_zero_tensor_is_explicitly_neutral(self) -> None:
        metrics = tensor_metrics_from_velocity_gradients(np.zeros((1, 3, 3)))

        self.assertAlmostEqual(metrics["Q"][0], 0.0)
        self.assertAlmostEqual(metrics["lambda2"][0], 0.0)
        self.assertAlmostEqual(metrics["omega_normalized"][0], 0.5)
        self.assertTrue(metrics["zero_tensor"][0])

    def test_q_omega_identity_holds_for_nonzero_finite_tensors(self) -> None:
        gradients = np.array(
            [
                [[1.0, -2.0, 0.5], [4.0, -1.5, 0.0], [0.25, 1.0, 0.75]],
                [[0.0, -1.0, 0.0], [3.0, 0.5, 0.0], [0.0, 0.0, -0.5]],
            ]
        )
        metrics = tensor_metrics_from_velocity_gradients(gradients)

        reconstructed_q = (metrics["omega_normalized"] - 0.5) * metrics["tensor_energy"]
        np.testing.assert_allclose(reconstructed_q, metrics["Q"], rtol=0.0, atol=1e-14)

    def test_nan_tensor_is_not_classified(self) -> None:
        gradients = np.array([np.full((3, 3), np.nan)])
        metrics = tensor_metrics_from_velocity_gradients(gradients)

        for key in ("S_norm2", "Omega_norm2", "tensor_energy", "Q", "lambda2", "omega_normalized"):
            self.assertTrue(np.isnan(metrics[key][0]))
        self.assertFalse(metrics["zero_tensor"][0])


if __name__ == "__main__":
    unittest.main()
