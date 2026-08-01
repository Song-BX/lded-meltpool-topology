from __future__ import annotations

import numpy as np


def tensor_metrics_from_velocity_gradients(gradients: np.ndarray) -> dict[str, np.ndarray]:
    """Return shared tensor invariants for finite 3-by-3 velocity gradients.

    ``omega_normalized`` uses the exact ratio of rotation to total tensor
    energy.  A finite zero tensor is explicitly neutral (0.5), rather than
    being assigned an arbitrary numerical offset or a rotation class.
    """
    tensor = np.asarray(gradients, dtype=float)
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        raise ValueError(f"Expected gradients with shape (n, 3, 3), received {tensor.shape}")

    finite = np.isfinite(tensor).all(axis=(1, 2))
    strain_norm2 = np.full(len(tensor), np.nan)
    rotation_norm2 = np.full(len(tensor), np.nan)
    tensor_energy = np.full(len(tensor), np.nan)
    q_values = np.full(len(tensor), np.nan)
    lambda2 = np.full(len(tensor), np.nan)
    omega_normalized = np.full(len(tensor), np.nan)
    zero_tensor = np.zeros(len(tensor), dtype=bool)
    if not finite.any():
        return {
            "S_norm2": strain_norm2,
            "Omega_norm2": rotation_norm2,
            "tensor_energy": tensor_energy,
            "Q": q_values,
            "lambda2": lambda2,
            "omega_normalized": omega_normalized,
            "zero_tensor": zero_tensor,
        }

    selected = tensor[finite]
    strain = 0.5 * (selected + np.swapaxes(selected, 1, 2))
    rotation = 0.5 * (selected - np.swapaxes(selected, 1, 2))
    strain_selected = np.einsum("nij,nij->n", strain, strain)
    rotation_selected = np.einsum("nij,nij->n", rotation, rotation)
    energy_selected = strain_selected + rotation_selected
    lambda2_matrix = strain @ strain + rotation @ rotation
    lambda2_matrix = 0.5 * (lambda2_matrix + np.swapaxes(lambda2_matrix, 1, 2))
    lambda2_selected = np.linalg.eigvalsh(lambda2_matrix)[:, 1]
    omega_selected = np.full(len(selected), 0.5)
    nonzero = energy_selected > 0
    omega_selected[nonzero] = rotation_selected[nonzero] / energy_selected[nonzero]

    strain_norm2[finite] = strain_selected
    rotation_norm2[finite] = rotation_selected
    tensor_energy[finite] = energy_selected
    q_values[finite] = 0.5 * (rotation_selected - strain_selected)
    lambda2[finite] = lambda2_selected
    omega_normalized[finite] = omega_selected
    zero_tensor[finite] = ~nonzero
    return {
        "S_norm2": strain_norm2,
        "Omega_norm2": rotation_norm2,
        "tensor_energy": tensor_energy,
        "Q": q_values,
        "lambda2": lambda2,
        "omega_normalized": omega_normalized,
        "zero_tensor": zero_tensor,
    }


def q_from_velocity_gradients(gradients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the legacy strain, rotation, and Q arrays without changing callers."""
    metrics = tensor_metrics_from_velocity_gradients(gradients)
    return metrics["S_norm2"], metrics["Omega_norm2"], metrics["Q"]
