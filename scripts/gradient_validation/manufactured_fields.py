from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.analysis.tensor_metrics import tensor_metrics_from_velocity_gradients

from .config import FieldSpec


RATE_S_INV = 1_000.0
VELOCITY_SCALE_MPS = 0.1


@dataclass(frozen=True)
class ManufacturedField:
    spec: FieldSpec
    frame: pd.DataFrame
    true_gradients: np.ndarray
    true_q: np.ndarray
    true_lambda2: np.ndarray
    true_omega_normalized: np.ndarray
    true_zero_tensor: np.ndarray


def _centred_coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    points = frame[["x", "y", "z"]].to_numpy(dtype=float)
    centre = np.median(points, axis=0)
    return points - centre, centre


def _affine_gradients(field_id: str, size: int) -> np.ndarray:
    gradient = np.zeros((size, 3, 3), dtype=float)
    if field_id == "affine_rotation":
        gradient[:, 0, 1] = -RATE_S_INV
        gradient[:, 1, 0] = RATE_S_INV
    elif field_id == "affine_strain":
        gradient[:, 0, 0] = RATE_S_INV
        gradient[:, 1, 1] = -RATE_S_INV
    elif field_id == "simple_shear_zero_q":
        gradient[:, 0, 2] = RATE_S_INV
    else:
        raise ValueError(f"Unknown affine manufactured field: {field_id}")
    return gradient


def _affine_velocity(field_id: str, offsets: np.ndarray) -> np.ndarray:
    if field_id == "affine_rotation":
        return np.column_stack(
            (-RATE_S_INV * offsets[:, 1], RATE_S_INV * offsets[:, 0], np.zeros(len(offsets)))
        )
    if field_id == "affine_strain":
        return np.column_stack(
            (RATE_S_INV * offsets[:, 0], -RATE_S_INV * offsets[:, 1], np.zeros(len(offsets)))
        )
    if field_id == "simple_shear_zero_q":
        return np.column_stack(
            (RATE_S_INV * offsets[:, 2], np.zeros(len(offsets)), np.zeros(len(offsets)))
        )
    raise ValueError(f"Unknown affine manufactured field: {field_id}")


def _gaussian_vortex(offsets: np.ndarray, scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = offsets.T
    radial2 = x**2 + y**2 + z**2
    envelope = np.exp(-radial2 / (2.0 * scale_m**2))
    omega = VELOCITY_SCALE_MPS / scale_m
    velocity = np.column_stack((-omega * z * envelope, np.zeros(len(x)), omega * x * envelope))
    gradient = np.zeros((len(x), 3, 3), dtype=float)
    gradient[:, 0, 0] = omega * x * z * envelope / scale_m**2
    gradient[:, 0, 2] = -omega * envelope * (1.0 - z**2 / scale_m**2)
    gradient[:, 2, 0] = omega * envelope * (1.0 - x**2 / scale_m**2)
    gradient[:, 2, 2] = -omega * x * z * envelope / scale_m**2
    return velocity, gradient


def _tanh_shear(offsets: np.ndarray, scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    normalized_z = offsets[:, 2] / scale_m
    hyperbolic_tangent = np.tanh(normalized_z)
    velocity = np.column_stack(
        (VELOCITY_SCALE_MPS * hyperbolic_tangent, np.zeros(len(offsets)), np.zeros(len(offsets)))
    )
    gradient = np.zeros((len(offsets), 3, 3), dtype=float)
    gradient[:, 0, 2] = (
        VELOCITY_SCALE_MPS / scale_m * (1.0 - hyperbolic_tangent**2)
    )
    return velocity, gradient


def build_manufactured_field(frame: pd.DataFrame, spec: FieldSpec) -> ManufacturedField:
    """Assign an analytically differentiable velocity field to observed coordinates."""
    offsets, _ = _centred_coordinates(frame)
    if spec.field_class == "affine":
        velocity = _affine_velocity(spec.field_id, offsets)
        gradients = _affine_gradients(spec.field_id, len(frame))
    elif spec.field_id == "gaussian_vortex" and spec.scale_m is not None:
        velocity, gradients = _gaussian_vortex(offsets, spec.scale_m)
    elif spec.field_id == "tanh_shear" and spec.scale_m is not None:
        velocity, gradients = _tanh_shear(offsets, spec.scale_m)
    else:
        raise ValueError(f"Unsupported manufactured field specification: {spec}")

    synthetic = frame.copy()
    synthetic[["u", "v", "w"]] = velocity
    synthetic["V"] = np.linalg.norm(velocity, axis=1)
    tensor_metrics = tensor_metrics_from_velocity_gradients(gradients)
    return ManufacturedField(
        spec=spec,
        frame=synthetic,
        true_gradients=gradients,
        true_q=tensor_metrics["Q"],
        true_lambda2=tensor_metrics["lambda2"],
        true_omega_normalized=tensor_metrics["omega_normalized"],
        true_zero_tensor=tensor_metrics["zero_tensor"],
    )
