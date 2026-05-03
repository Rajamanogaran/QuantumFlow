"""
Quantum Error Mitigation Techniques
====================================

Methods for reducing the impact of noise on quantum computation results
without requiring full quantum error correction.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ErrorMitigation(ABC):
    """Abstract base class for error mitigation techniques."""

    @abstractmethod
    def mitigate(self, counts: Dict[str, int], **kwargs) -> Dict[str, float]:
        """Apply error mitigation to measurement results."""
        pass

    @abstractmethod
    def calibrate(self, **kwargs) -> None:
        """Calibrate the error mitigation model."""
        pass


class ZeroNoiseExtrapolation(ErrorMitigation):
    """
    Zero Noise Extrapolation (ZNE).

    Estimates the noiseless result by running the circuit at multiple
    noise levels and extrapolating to zero noise.

    Methods:
    - Richardson: polynomial extrapolation
    - Exponential: exponential decay fitting
    - Linear: linear regression

    Parameters
    ----------
    noise_factors : List[float]
        Scale factors for noise (e.g., [1.0, 2.0, 3.0]).
    method : str
        Extrapolation method: 'richardson', 'exponential', 'linear'.

    Examples
    --------
    >>> zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0])
    >>> zne.calibrate(base_noise=0.01)
    >>> mitigated = zne.mitigate(counts)
    """

    def __init__(
        self,
        noise_factors: Optional[List[float]] = None,
        method: str = 'richardson',
    ) -> None:
        self.noise_factors = noise_factors or [1.0, 2.0, 3.0]
        self.method = method
        self._noisy_results: List[float] = []

    def calibrate(self, base_noise: float = 0.01, **kwargs) -> None:
        """Store calibration parameters."""
        self.base_noise = base_noise

    def mitigate(
        self,
        counts: Dict[str, int],
        expectation_fn: Optional[Callable] = None,
        noisy_expectations: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Perform zero noise extrapolation.

        Parameters
        ----------
        counts : Dict[str, int]
            Measurement counts from the unmodified circuit.
        expectation_fn : Optional[Callable]
            Function to compute expectation value from counts.
        noisy_expectations : Optional[List[float]]
            Pre-computed expectation values at each noise factor.

        Returns
        -------
        Dict[str, Any]
            Mitigated expectation value and metadata.
        """
        if noisy_expectations is not None:
            self._noisy_results = noisy_expectations
        elif expectation_fn is not None:
            # Would need to run circuits at each noise factor
            self._noisy_results = [expectation_fn(counts)]

        if len(self._noisy_results) < 2:
            return {'mitigated_value': self._noisy_results[0] if self._noisy_results else 0.0}

        if self.method == 'richardson':
            mitigated = self._richardson()
        elif self.method == 'exponential':
            mitigated = self._exponential()
        elif self.method == 'linear':
            mitigated = self._linear()
        else:
            mitigated = self._richardson()

        return {
            'mitigated_value': mitigated,
            'noisy_values': self._noisy_results,
            'noise_factors': self.noise_factors,
            'method': self.method,
        }

    def _richardson(self) -> float:
        """Richardson extrapolation (polynomial)."""
        n = len(self.noise_factors)
        if n < 2:
            return self._noisy_results[0]

        # Richardson extrapolation weights
        # For factors [1, 2, 3]: w_i = product_{j!=i} 1/(x_i - x_j)
        x = np.array(self.noise_factors)
        y = np.array(self._noisy_results[:n])

        weights = np.zeros(n)
        for i in range(n):
            w = 1.0
            for j in range(n):
                if i != j:
                    w *= 1.0 / (x[i] - x[j])
            weights[i] = w

        return float(np.dot(y, weights))

    def _exponential(self) -> float:
        """Exponential extrapolation."""
        x = np.array(self.noise_factors)
        y = np.array(self._noisy_results[:len(x)])

        # Fit y = a * exp(-b*x) + c using least squares
        try:
            # Initial guess
            a0 = y[0] - y[-1]
            b0 = 1.0
            c0 = y[-1]

            # Simple grid search for best fit
            best_cost = float('inf')
            best_params = (a0, b0, c0)

            for a_try in np.linspace(a0 * 0.5, a0 * 2.0, 20):
                for b_try in np.linspace(0.1, 5.0, 20):
                    for c_try in np.linspace(c0 * 0.5, c0 * 2.0, 20):
                        y_pred = a_try * np.exp(-b_try * x) + c_try
                        cost = np.sum((y - y_pred) ** 2)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = (a_try, b_try, c_try)

            a, b, c = best_params
            return float(a * np.exp(0) + c)  # x=0
        except Exception:
            return self._linear()

    def _linear(self) -> float:
        """Linear extrapolation to zero noise."""
        x = np.array(self.noise_factors)
        y = np.array(self._noisy_results[:len(x)])

        # Linear fit: y = a*x + b, extrapolate to x=0
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            return float(np.polyval(coeffs, 0))
        return float(y[0])


class ProbabilisticErrorCancellation(ErrorMitigation):
    """
    Probabilistic Error Cancellation (PEC).

    Decomposes noisy operations into a linear combination of implementable
    operations, then uses sampling to estimate the noiseless result.

    Parameters
    ----------
    n_samples : int
        Number of PEC samples.
    """

    def __init__(self, n_samples: int = 1000) -> None:
        self.n_samples = n_samples
        self._inverse_noise: Optional[np.ndarray] = None

    def calibrate(
        self,
        noise_channel: Optional[np.ndarray] = None,
        gate_inverse: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Calibrate by computing the inverse noise operation."""
        if noise_channel is not None:
            self._noise_channel = noise_channel
            # Compute quasi-inverse (pseudo-inverse)
            self._inverse_noise = np.linalg.pinv(noise_channel)
        elif gate_inverse is not None:
            self._inverse_noise = gate_inverse

    def mitigate(
        self,
        counts: Dict[str, int],
        noise_channel: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Apply probabilistic error cancellation.

        Parameters
        ----------
        counts : Dict[str, int]
            Noisy measurement counts.
        noise_channel : Optional[np.ndarray]
            Noise superoperator matrix.

        Returns
        -------
        Dict[str, Any]
            Mitigated counts and statistics.
        """
        total = sum(counts.values())
        if total == 0:
            return {'mitigated_counts': counts, 'sign': 1.0}

        if noise_channel is not None:
            self.calibrate(noise_channel=noise_channel)

        if self._inverse_noise is None:
            return {'mitigated_counts': counts, 'sign': 1.0}

        # Apply inverse noise to probability distribution
        probs = np.array([counts.get(format(i, f'0{int(np.log2(total)) if total > 0 else 1}b'), 0) / total
                         for i in range(len(counts))])
        probs = probs / probs.sum() if probs.sum() > 0 else probs

        try:
            mitigated_probs = self._inverse_noise @ probs
            # Ensure non-negative
            mitigated_probs = np.maximum(mitigated_probs, 0)
            mitigated_probs = mitigated_probs / mitigated_probs.sum()

            mitigated_counts = {}
            for i, (bitstring, count) in enumerate(counts.items()):
                mitigated_counts[bitstring] = int(mitigated_probs[i] * total)

            return {
                'mitigated_counts': mitigated_counts,
                'sign': float(np.sum(np.abs(mitigated_probs))),
                'n_samples': self.n_samples,
            }
        except Exception:
            return {'mitigated_counts': counts, 'sign': 1.0}


class MeasurementErrorMitigation(ErrorMitigation):
    """
    Measurement Error Mitigation.

    Corrects for readout errors by calibrating a confusion matrix
    and applying its inverse to measurement results.

    Parameters
    ----------
    n_qubits : int
        Number of measured qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._confusion_matrix: Optional[np.ndarray] = None
        self._inverse_matrix: Optional[np.ndarray] = None
        self._n_outcomes = 2 ** n_qubits

    def calibrate(
        self,
        confusion_matrix: Optional[np.ndarray] = None,
        calibration_circuits: Optional[List[Tuple[str, Dict[str, int]]]] = None,
    ) -> None:
        """
        Calibrate the confusion matrix.

        Parameters
        ----------
        confusion_matrix : Optional[np.ndarray]
            Pre-computed confusion matrix (n x n).
        calibration_circuits : Optional[List[Tuple[str, Dict[str, int]]]]
            List of (prepared_state, measured_counts) from calibration runs.
        """
        if confusion_matrix is not None:
            self._confusion_matrix = confusion_matrix
        elif calibration_circuits is not None:
            n = self._n_outcomes
            self._confusion_matrix = np.zeros((n, n), dtype=np.float64)

            for bitstring, counts in calibration_circuits:
                row = int(bitstring, 2)
                total = sum(counts.values())
                for outcome, count in counts.items():
                    col = int(outcome, 2)
                    self._confusion_matrix[row, col] = count / total

            # Regularize
            self._confusion_matrix += 1e-10 * np.eye(n)

        if self._confusion_matrix is not None:
            # Compute inverse with regularization
            self._inverse_matrix = np.linalg.pinv(self._confusion_matrix)

    def mitigate(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Apply measurement error mitigation.

        Parameters
        ----------
        counts : Dict[str, int]
            Noisy measurement counts.

        Returns
        -------
        Dict[str, Any]
            Mitigated counts.
        """
        if self._inverse_matrix is None:
            return {'mitigated_counts': counts, 'improvement': 0.0}

        n = self._n_outcomes

        # Build probability vector
        noisy_probs = np.zeros(n, dtype=np.float64)
        for bitstring, count in counts.items():
            if len(bitstring) <= self.n_qubits:
                idx = int(bitstring, 2)
                noisy_probs[idx] = count

        total = noisy_probs.sum()
        if total > 0:
            noisy_probs = noisy_probs / total

        # Apply inverse confusion matrix
        mitigated_probs = self._inverse_matrix @ noisy_probs
        mitigated_probs = np.maximum(mitigated_probs, 0)
        prob_sum = mitigated_probs.sum()
        if prob_sum > 0:
            mitigated_probs = mitigated_probs / prob_sum

        mitigated_counts = {}
        for i in range(n):
            bitstring = format(i, f'0{self.n_qubits}b')
            mitigated_counts[bitstring] = int(mitigated_probs[i] * total)

        # Compute improvement
        identity_prob = np.eye(n)[np.argmax(noisy_probs)].sum()
        mitigated_max = mitigated_probs.max()

        return {
            'mitigated_counts': mitigated_counts,
            'mitigated_probs': mitigated_probs,
            'improvement': float(mitigated_max - identity_prob),
            'confusion_matrix': self._confusion_matrix,
        }

    @classmethod
    def create_confusion_matrix(
        cls,
        n_qubits: int,
        assignment_probs: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """
        Create a confusion matrix from per-qubit assignment probabilities.

        Parameters
        ----------
        n_qubits : int
        assignment_probs : Optional[Dict[int, float]]
            Per-qubit readout error probability (P(read 1|prepared 0)).

        Returns
        -------
        np.ndarray
            Confusion matrix.
        """
        n = 2 ** n_qubits
        cm = np.ones((n, n), dtype=np.float64)

        if assignment_probs is None:
            assignment_probs = {i: 0.01 for i in range(n_qubits)}

        for i in range(n):
            for j in range(n):
                prob = 1.0
                for q in range(n_qubits):
                    bit_i = (i >> q) & 1
                    bit_j = (j >> q) & 1
                    p = assignment_probs.get(q, 0.01)
                    if bit_i == bit_j:
                        prob *= (1 - p)
                    else:
                        prob *= p
                cm[i, j] = prob

        return cm


class VirtualDistillation(ErrorMitigation):
    """
    Virtual Distillation.

    Purifies a noisy state by computing rho^k / Tr(rho^k).
    For k=2, this squares the density matrix, amplifying the
    dominant eigenstate.

    Parameters
    ----------
    power : int
        Distillation power (k). Higher = more purification but noisier.
    """

    def __init__(self, power: int = 2) -> None:
        self.power = power

    def calibrate(self, **kwargs) -> None:
        """No calibration needed for virtual distillation."""
        pass

    def mitigate(
        self,
        rho: Optional[np.ndarray] = None,
        counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply virtual distillation.

        Parameters
        ----------
        rho : Optional[np.ndarray]
            Density matrix (preferred).
        counts : Optional[Dict[str, int]]
            Measurement counts (converted to density matrix).

        Returns
        -------
        Dict[str, Any]
            Purified density matrix or mitigated counts.
        """
        if rho is not None:
            purified = rho
            for _ in range(self.power - 1):
                purified = purified @ purified
            trace = np.trace(purified).real
            if trace > 0:
                purified = purified / trace
            return {
                'purified_state': purified,
                'purity': float(np.trace(purified @ purified).real),
                'method': f'virtual_distillation_k={self.power}',
            }

        if counts is not None:
            # Convert counts to density matrix
            total = sum(counts.values())
            n_outcomes = len(counts)
            rho = np.zeros((n_outcomes, n_outcomes), dtype=np.complex128)
            for bs, count in counts.items():
                idx = int(bs, 2)
                rho[idx, idx] = count / total
            return self.mitigate(rho=rho)

        return {'error': 'Provide either rho or counts'}

    def purified_expectation(
        self,
        rho: np.ndarray,
        observable: np.ndarray,
    ) -> float:
        """
        Compute purified expectation value without full state tomography.

        <O>_k = Tr(O * rho^k) / Tr(rho^k)

        This can be computed more efficiently using circuit duplication.
        """
        rho_k = np.linalg.matrix_power(rho, self.power)
        numerator = np.trace(observable @ rho_k).real
        denominator = np.trace(rho_k).real
        return float(numerator / denominator) if denominator > 0 else 0.0


class SymmetryVerification(ErrorMitigation):
    """
    Symmetry-based error mitigation.

    Discards measurement outcomes that violate known symmetries
    of the system (e.g., particle number conservation).

    Parameters
    ----------
    symmetry_check : Optional[Callable]
        Function that returns True if a bitstring satisfies the symmetry.
    """

    def __init__(self, symmetry_check: Optional[Callable] = None) -> None:
        self.symmetry_check = symmetry_check

    def calibrate(self, symmetry_check: Optional[Callable] = None, **kwargs) -> None:
        if symmetry_check is not None:
            self.symmetry_check = symmetry_check

    def mitigate(self, counts: Dict[str, int], **kwargs) -> Dict[str, Any]:
        """
        Filter measurement results by symmetry.

        Parameters
        ----------
        counts : Dict[str, int]
            Raw measurement counts.

        Returns
        -------
        Dict[str, Any]
            Filtered counts and rejection rate.
        """
        if self.symmetry_check is None:
            return {'mitigated_counts': counts, 'rejection_rate': 0.0}

        filtered = {}
        rejected = 0

        for bitstring, count in counts.items():
            if self.symmetry_check(bitstring):
                filtered[bitstring] = count
            else:
                rejected += count

        total = sum(counts.values())
        rejection_rate = rejected / total if total > 0 else 0.0

        return {
            'mitigated_counts': filtered,
            'rejection_rate': rejection_rate,
            'n_rejected': rejected,
        }
