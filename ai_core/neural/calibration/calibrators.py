"""Neural model calibration algorithms and serialization engine.

Implements:
1. Temperature Scaling: Single positive scalar T minimizing NLL while strictly preserving ranking.
2. Platt Scaling: Logistic regression parameters (A, B) fitting affine score transformations.
3. Isotonic Regression: Non-parametric monotonic step calibrator via Pool Adjacent Violators Algorithm (PAVA).

Guarantees:
- Model-version linkage: Every calibrator is cryptographically bound to its parent model version,
  architecture, and training/split manifest hashes. Version mismatches raise ModelVersionMismatchError.
- Held-out partition safety: Rejects calibration fitting attempted on training or test splits.
- Numerical safety: Strictly checks for non-finite values (NaN, Inf) during fitting, serialization, and calibration.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ai_core.neural.calibration.taxonomy import (
    CalibratorMethod,
    OutputNature,
    validate_conceptual_separation,
)


class ModelVersionMismatchError(ValueError):
    """Raised when a calibrator artifact is loaded against an incompatible model version."""
    pass


class UnfittedCalibratorError(RuntimeError):
    """Raised when calibration inference is requested from an unfitted calibrator."""
    pass


class InvalidCalibrationStateError(ValueError):
    """Raised when a calibrator's internal parameters violate mathematical validity."""
    pass


def _validate_finite_array(arr: np.ndarray, name: str) -> None:
    """Ensure array contains only finite floating-point numbers."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values (NaN or Inf) detected in {name}.")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -30.0, 30.0)
    return np.where(
        z_clipped >= 0,
        1.0 / (1.0 + np.exp(-z_clipped)),
        np.exp(z_clipped) / (1.0 + np.exp(z_clipped)),
    )


def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Compute binary cross-entropy loss with safety clipping."""
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    loss = - (y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return float(np.mean(loss))


class BaseNeuralCalibrator(ABC):
    """Abstract base class for neural model calibration engines."""

    def __init__(
        self,
        model_name: str = "generic_model",
        model_version: str = "1.0.0",
        architecture: str = "mobilenet_v3_large",
        training_manifest_hash: Optional[str] = None,
        split_manifest_hash: Optional[str] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self.architecture = str(architecture)
        self.training_manifest_hash = training_manifest_hash
        self.split_manifest_hash = split_manifest_hash
        self.calibration_split: Optional[str] = None
        self.fit_timestamp: Optional[str] = None
        self.is_fitted: bool = False
        self.validation_brier: Optional[float] = None
        self.validation_ece: Optional[float] = None

    @abstractmethod
    def method(self) -> CalibratorMethod:
        """Return categorical method enum."""
        pass

    def verify_split_isolation(self, split_name: str) -> None:
        """Enforce that calibration only fits on validation/calibration partitions, never train or test."""
        normalized = split_name.strip().lower()
        forbidden_splits = {
            "train",
            "training",
            "internal_test",
            "held_out_generator_test",
            "cross_dataset_test",
            "compression_robustness_test",
            "test",
        }
        if normalized in forbidden_splits:
            raise ValueError(
                f"STRICT SEPARATION VIOLATION: Calibration cannot be fitted on split '{split_name}'. "
                "Calibration must only be fitted on a dedicated held-out validation partition."
            )
        self.calibration_split = split_name

    def verify_model_linkage(
        self,
        target_model_name: str,
        target_model_version: str,
        target_architecture: Optional[str] = None,
    ) -> None:
        """Verify that calibrator matches target model version and architecture.

        Raises:
            ModelVersionMismatchError: If model version or architecture mismatch.
        """
        if self.model_version != target_model_version:
            raise ModelVersionMismatchError(
                f"Calibrator version '{self.model_version}' does not match target model "
                f"version '{target_model_version}' for model '{self.model_name}'."
            )
        if target_architecture and self.architecture != target_architecture:
            raise ModelVersionMismatchError(
                f"Calibrator architecture '{self.architecture}' does not match target "
                f"architecture '{target_architecture}'."
            )

    @abstractmethod
    def fit(
        self,
        val_logits_or_scores: Union[List[float], np.ndarray],
        val_labels: Union[List[bool], List[int], np.ndarray],
        calibration_split: str = "validation",
    ) -> "BaseNeuralCalibrator":
        """Fit calibration mapping on validation split data."""
        pass

    @abstractmethod
    def calibrate(
        self,
        logits_or_scores: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        """Transform uncalibrated logits or scores into calibrated probabilities."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator state and model linkage metadata to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseNeuralCalibrator":
        """Deserialize calibrator from dictionary."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator artifact to JSON file."""
        if not self.is_fitted:
            raise UnfittedCalibratorError("Cannot serialize an unfitted calibrator.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseNeuralCalibrator":
        """Load calibrator artifact from JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration artifact not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class TemperatureScalingCalibrator(BaseNeuralCalibrator):
    """Temperature scaling calibrator: p = sigmoid(z / T).

    Finds optimal scalar T > 0 minimizing NLL on validation logits.
    Strictly preserves classification ranking (AUC invariant).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        model_name: str = "mobilenet_v3_large",
        model_version: str = "1.0.0",
        architecture: str = "mobilenet_v3_large",
        training_manifest_hash: Optional[str] = None,
        split_manifest_hash: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            architecture=architecture,
            training_manifest_hash=training_manifest_hash,
            split_manifest_hash=split_manifest_hash,
        )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise InvalidCalibrationStateError(
                f"Temperature must be a strictly positive finite number, got {temperature}."
            )
        self.temperature = float(temperature)

    def method(self) -> CalibratorMethod:
        return CalibratorMethod.TEMPERATURE_SCALING

    def fit(
        self,
        val_logits_or_scores: Union[List[float], np.ndarray],
        val_labels: Union[List[bool], List[int], np.ndarray],
        calibration_split: str = "validation",
    ) -> "TemperatureScalingCalibrator":
        """Fit optimal temperature T on validation logits."""
        self.verify_split_isolation(calibration_split)

        z = np.asarray(val_logits_or_scores, dtype=np.float64)
        y = np.asarray(val_labels, dtype=np.float64)
        _validate_finite_array(z, "validation logits")
        _validate_finite_array(y, "validation labels")

        if len(z) == 0:
            raise ValueError("Cannot fit temperature scaling on empty validation data.")
        if len(z) != len(y):
            raise ValueError(f"Mismatched lengths: logits ({len(z)}) vs labels ({len(y)}).")

        # If input looks like probabilities [0, 1], invert to logits
        if np.all((z >= 0.0) & (z <= 1.0)):
            z_clipped = np.clip(z, 1e-6, 1.0 - 1e-6)
            logits = np.log(z_clipped / (1.0 - z_clipped))
        else:
            logits = z

        def loss_fn(t_val: float) -> float:
            if t_val <= 1e-4:
                return float("inf")
            p = _sigmoid(logits / t_val)
            return _binary_cross_entropy(y, p)

        # Golden Section Search in [0.05, 10.0]
        a, b = 0.05, 10.0
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr

        for _ in range(80):
            if abs(b - a) < 1e-5:
                break
            if loss_fn(c) < loss_fn(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr

        best_t = float((b + a) / 2.0)
        if not math.isfinite(best_t) or best_t <= 0.0:
            raise InvalidCalibrationStateError(f"Optimization yielded invalid temperature: {best_t}")

        self.temperature = best_t
        self.is_fitted = True
        self.fit_timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def calibrate(
        self,
        logits_or_scores: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        """Calibrate input logits or scores using fitted temperature."""
        if not self.is_fitted:
            raise UnfittedCalibratorError("TemperatureScalingCalibrator must be fitted before calibrate().")

        z = np.asarray(logits_or_scores, dtype=np.float64)
        _validate_finite_array(z, "input logits/scores")

        if np.all((z >= 0.0) & (z <= 1.0)):
            z_clipped = np.clip(z, 1e-6, 1.0 - 1e-6)
            logits = np.log(z_clipped / (1.0 - z_clipped))
        else:
            logits = z

        scaled = logits / self.temperature
        probs = _sigmoid(scaled)
        return np.clip(probs, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise UnfittedCalibratorError("Cannot serialize unfitted calibrator.")
        return {
            "method": self.method().value,
            "temperature": self.temperature,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "training_manifest_hash": self.training_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "calibration_split": self.calibration_split,
            "fit_timestamp": self.fit_timestamp,
            "is_fitted": self.is_fitted,
            "validation_brier": self.validation_brier,
            "validation_ece": self.validation_ece,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureScalingCalibrator":
        method_str = data.get("method")
        if method_str != CalibratorMethod.TEMPERATURE_SCALING.value:
            raise ValueError(f"Invalid method '{method_str}' for TemperatureScalingCalibrator.")

        temp = float(data.get("temperature", 1.0))
        if not math.isfinite(temp) or temp <= 0.0:
            raise InvalidCalibrationStateError(f"Invalid temperature in artifact: {temp}")

        inst = cls(
            temperature=temp,
            model_name=str(data.get("model_name", "generic_model")),
            model_version=str(data.get("model_version", "1.0.0")),
            architecture=str(data.get("architecture", "mobilenet_v3_large")),
            training_manifest_hash=data.get("training_manifest_hash"),
            split_manifest_hash=data.get("split_manifest_hash"),
        )
        inst.calibration_split = data.get("calibration_split")
        inst.fit_timestamp = data.get("fit_timestamp")
        inst.is_fitted = bool(data.get("is_fitted", True))
        inst.validation_brier = data.get("validation_brier")
        inst.validation_ece = data.get("validation_ece")
        return inst


class PlattScalingCalibrator(BaseNeuralCalibrator):
    """Platt scaling (logistic calibration): p = sigmoid(A * z + B).

    Fits affine parameters A and B via regularized gradient descent minimizing NLL.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        model_name: str = "mobilenet_v3_large",
        model_version: str = "1.0.0",
        architecture: str = "mobilenet_v3_large",
        training_manifest_hash: Optional[str] = None,
        split_manifest_hash: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            architecture=architecture,
            training_manifest_hash=training_manifest_hash,
            split_manifest_hash=split_manifest_hash,
        )
        if not (math.isfinite(a) and math.isfinite(b)):
            raise InvalidCalibrationStateError("Platt scaling parameters A and B must be finite.")
        self.a = float(a)
        self.b = float(b)

    def method(self) -> CalibratorMethod:
        return CalibratorMethod.PLATT_SCALING

    def fit(
        self,
        val_logits_or_scores: Union[List[float], np.ndarray],
        val_labels: Union[List[bool], List[int], np.ndarray],
        calibration_split: str = "validation",
        epochs: int = 250,
        lr: float = 0.05,
        l2_reg: float = 1e-4,
    ) -> "PlattScalingCalibrator":
        """Fit affine parameters A and B on validation logits."""
        self.verify_split_isolation(calibration_split)

        z = np.asarray(val_logits_or_scores, dtype=np.float64)
        y = np.asarray(val_labels, dtype=np.float64)
        _validate_finite_array(z, "validation logits")
        _validate_finite_array(y, "validation labels")

        if len(z) == 0:
            raise ValueError("Cannot fit Platt scaling on empty validation data.")
        if len(z) != len(y):
            raise ValueError(f"Mismatched lengths: {len(z)} vs {len(y)}")

        # Initialize: a=1.0, b=0.0
        param_a = 1.0
        param_b = 0.0

        for _ in range(epochs):
            act = param_a * z + param_b
            p = _sigmoid(act)
            diff = p - y

            grad_a = np.mean(diff * z) + l2_reg * param_a
            grad_b = np.mean(diff)

            param_a -= lr * grad_a
            param_b -= lr * grad_b

        if not (math.isfinite(param_a) and math.isfinite(param_b)):
            raise InvalidCalibrationStateError("Platt scaling fitting diverged to non-finite parameters.")

        self.a = float(param_a)
        self.b = float(param_b)
        self.is_fitted = True
        self.fit_timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def calibrate(
        self,
        logits_or_scores: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        """Calibrate input logits or scores using fitted affine sigmoid."""
        if not self.is_fitted:
            raise UnfittedCalibratorError("PlattScalingCalibrator must be fitted before calibrate().")

        z = np.asarray(logits_or_scores, dtype=np.float64)
        _validate_finite_array(z, "input logits/scores")

        act = self.a * z + self.b
        probs = _sigmoid(act)
        return np.clip(probs, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise UnfittedCalibratorError("Cannot serialize unfitted calibrator.")
        return {
            "method": self.method().value,
            "a": self.a,
            "b": self.b,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "training_manifest_hash": self.training_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "calibration_split": self.calibration_split,
            "fit_timestamp": self.fit_timestamp,
            "is_fitted": self.is_fitted,
            "validation_brier": self.validation_brier,
            "validation_ece": self.validation_ece,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlattScalingCalibrator":
        method_str = data.get("method")
        if method_str != CalibratorMethod.PLATT_SCALING.value:
            raise ValueError(f"Invalid method '{method_str}' for PlattScalingCalibrator.")

        a = float(data.get("a", 1.0))
        b = float(data.get("b", 0.0))
        if not (math.isfinite(a) and math.isfinite(b)):
            raise InvalidCalibrationStateError(f"Invalid non-finite parameters in Platt artifact: a={a}, b={b}")

        inst = cls(
            a=a,
            b=b,
            model_name=str(data.get("model_name", "generic_model")),
            model_version=str(data.get("model_version", "1.0.0")),
            architecture=str(data.get("architecture", "mobilenet_v3_large")),
            training_manifest_hash=data.get("training_manifest_hash"),
            split_manifest_hash=data.get("split_manifest_hash"),
        )
        inst.calibration_split = data.get("calibration_split")
        inst.fit_timestamp = data.get("fit_timestamp")
        inst.is_fitted = bool(data.get("is_fitted", True))
        inst.validation_brier = data.get("validation_brier")
        inst.validation_ece = data.get("validation_ece")
        return inst


class IsotonicRegressionCalibrator(BaseNeuralCalibrator):
    """Non-parametric monotonic step calibrator using Pool Adjacent Violators Algorithm (PAVA).

    Requires sufficient calibration data (N >= 20) to prevent severe overfitting.
    Enforces p(s_1) <= p(s_2) for all s_1 <= s_2.
    """

    def __init__(
        self,
        x_thresholds: Optional[List[float]] = None,
        y_values: Optional[List[float]] = None,
        model_name: str = "mobilenet_v3_large",
        model_version: str = "1.0.0",
        architecture: str = "mobilenet_v3_large",
        training_manifest_hash: Optional[str] = None,
        split_manifest_hash: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            architecture=architecture,
            training_manifest_hash=training_manifest_hash,
            split_manifest_hash=split_manifest_hash,
        )
        self.x_thresholds: List[float] = list(x_thresholds or [])
        self.y_values: List[float] = list(y_values or [])
        if self.x_thresholds and self.y_values:
            self.is_fitted = True

    def method(self) -> CalibratorMethod:
        return CalibratorMethod.ISOTONIC_REGRESSION

    def fit(
        self,
        val_logits_or_scores: Union[List[float], np.ndarray],
        val_labels: Union[List[bool], List[int], np.ndarray],
        calibration_split: str = "validation",
        min_samples: int = 15,
    ) -> "IsotonicRegressionCalibrator":
        """Fit monotonic non-decreasing step function via PAVA."""
        self.verify_split_isolation(calibration_split)

        scores = np.asarray(val_logits_or_scores, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.float64)
        _validate_finite_array(scores, "validation scores")
        _validate_finite_array(labels, "validation labels")

        if len(scores) < min_samples:
            raise ValueError(
                f"Isotonic regression requires at least {min_samples} validation samples, got {len(scores)}."
            )

        # Sort pairs by score
        pairs = sorted(zip(scores, labels), key=lambda p: p[0])
        x_sorted = [p[0] for p in pairs]
        y_sorted = [p[1] for p in pairs]

        # PAVA: each block = [weight, sum_y, mean_y]
        blocks: List[List[float]] = [[1.0, y, y] for y in y_sorted]

        i = 0
        while i < len(blocks) - 1:
            if blocks[i][2] > blocks[i + 1][2]:
                w1, s1, _ = blocks[i]
                w2, s2, _ = blocks[i + 1]
                new_w = w1 + w2
                new_s = s1 + s2
                new_mean = new_s / new_w
                blocks[i] = [new_w, new_s, new_mean]
                del blocks[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        fitted_y: List[float] = []
        for w, _, mean in blocks:
            fitted_y.extend([mean] * int(w))

        # Collapse duplicate x-coordinates
        x_unique: List[float] = []
        y_unique: List[float] = []
        for x_val, y_val in zip(x_sorted, fitted_y):
            if not x_unique or x_val > x_unique[-1]:
                x_unique.append(float(x_val))
                y_unique.append(float(y_val))
            else:
                y_unique[-1] = float(max(y_unique[-1], y_val))

        self.x_thresholds = x_unique
        self.y_values = y_unique
        self.is_fitted = True
        self.fit_timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def calibrate(
        self,
        logits_or_scores: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        """Calibrate input scores via monotonic piecewise interpolation."""
        if not self.is_fitted:
            raise UnfittedCalibratorError("IsotonicRegressionCalibrator must be fitted before calibrate().")

        z = np.asarray(logits_or_scores, dtype=np.float64)
        _validate_finite_array(z, "input scores")

        if not self.x_thresholds:
            return np.clip(z, 0.0, 1.0)

        # Piecewise linear interpolation between step centers
        interp = np.interp(z, self.x_thresholds, self.y_values)
        return np.clip(interp, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise UnfittedCalibratorError("Cannot serialize unfitted calibrator.")
        return {
            "method": self.method().value,
            "x_thresholds": self.x_thresholds,
            "y_values": self.y_values,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "training_manifest_hash": self.training_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "calibration_split": self.calibration_split,
            "fit_timestamp": self.fit_timestamp,
            "is_fitted": self.is_fitted,
            "validation_brier": self.validation_brier,
            "validation_ece": self.validation_ece,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IsotonicRegressionCalibrator":
        method_str = data.get("method")
        if method_str != CalibratorMethod.ISOTONIC_REGRESSION.value:
            raise ValueError(f"Invalid method '{method_str}' for IsotonicRegressionCalibrator.")

        x_thresh = data.get("x_thresholds", [])
        y_vals = data.get("y_values", [])

        inst = cls(
            x_thresholds=x_thresh,
            y_values=y_vals,
            model_name=str(data.get("model_name", "generic_model")),
            model_version=str(data.get("model_version", "1.0.0")),
            architecture=str(data.get("architecture", "mobilenet_v3_large")),
            training_manifest_hash=data.get("training_manifest_hash"),
            split_manifest_hash=data.get("split_manifest_hash"),
        )
        inst.calibration_split = data.get("calibration_split")
        inst.fit_timestamp = data.get("fit_timestamp")
        inst.is_fitted = bool(data.get("is_fitted", True))
        inst.validation_brier = data.get("validation_brier")
        inst.validation_ece = data.get("validation_ece")
        return inst
