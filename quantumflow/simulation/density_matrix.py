"""
Density Matrix Simulation Backend
===================================

Full density-matrix quantum circuit simulator that tracks the complete
 ``(2ⁿ, 2ⁿ)`` density matrix ``ρ`` for an ``n``-qubit system.

The :class:`DensityMatrixBackend` supports:

* Unitary gate application via ``ρ → U ρ U†``.
* General quantum operations via Kraus operators
  ``ρ → Σₖ Eₖ ρ Eₖ†``.
* POVM measurement with state collapse.
* Partial trace for reducing the density matrix to a sub-system.
* Built-in noise channels (depolarising, amplitude damping, phase
  damping, bit-flip, phase-flip).
* Efficient tensor-contraction implementation.

Numerical stability
-------------------
The trace of the density matrix is renormalised to 1 after each
non-unitary operation to prevent drift.

Typical usage::

    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.simulation.density_matrix import DensityMatrixBackend

    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)

    backend = DensityMatrixBackend()
    rho = backend.run_circuit(qc)
    print(backend.purity(rho))
"""

from __future__ import annotations

import math
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import Gate, Measurement, ParameterizedGate
from quantumflow.core.operation import Barrier, Operation, Reset
from quantumflow.core.state import DensityMatrix as CoreDensityMatrix
from quantumflow.core.state import Statevector

__all__ = [
    "DensityMatrixBackend",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEX_DTYPE = np.complex128
_FLOAT_DTYPE = np.float64
_TOLERANCE = 1e-12
_EPS = 1e-15


# ---------------------------------------------------------------------------
# Noise channel helpers
# ---------------------------------------------------------------------------

class NoiseChannel:
    """Descriptor for a quantum noise channel.

    Parameters
    ----------
    name : str
        Channel identifier (e.g. ``'depolarizing'``, ``'amplitude_damping'``).
    params : tuple of float
        Channel parameters (e.g. error probability).
    """

    def __init__(self, name: str, params: Tuple[float, ...] = ()) -> None:
        self.name = name
        self.params = params

    def kraus_operators(self, num_qubits: int) -> List[np.ndarray]:
        """Return the list of Kraus operators for this channel.

        Parameters
        ----------
        num_qubits : int
            Number of qubits the channel acts on.

        Returns
        -------
        list of numpy.ndarray
        """
        ops = _NOISE_CHANNEL_REGISTRY.get(self.name)
        if ops is None:
            raise ValueError(
                f"Unknown noise channel '{self.name}'. "
                f"Available: {list(_NOISE_CHANNEL_REGISTRY.keys())}"
            )
        return ops(self.params, num_qubits)


def _depolarizing_kraus(
    params: Tuple[float, ...],
    num_qubits: int,
) -> List[np.ndarray]:
    """Depolarising channel: with probability *p* the state is replaced
    by the maximally mixed state.

    Kraus operators::

        K₀ = √(1 − p) I
        Kᵢ = √(p / 3) σᵢ   for i = 1, 2, 3  (single-qubit)

    For multi-qubit depolarising, *p* is applied per-qubit.
    """
    p = params[0] if params else 0.0
    dim = 1 << num_qubits

    if num_qubits == 1:
        I2 = np.eye(2, dtype=_COMPLEX_DTYPE)
        X = np.array([[0, 1], [1, 0]], dtype=_COMPLEX_DTYPE)
        Y = np.array([[0, -1j], [1j, 0]], dtype=_COMPLEX_DTYPE)
        Z = np.array([[1, 0], [0, -1]], dtype=_COMPLEX_DTYPE)

        k0 = math.sqrt(max(0.0, 1.0 - p)) * I2
        coeff = math.sqrt(max(0.0, p / 3.0))
        return [k0, coeff * X, coeff * Y, coeff * Z]
    else:
        # Multi-qubit: tensor product of single-qubit depolarising
        single_ops = _depolarizing_kraus(params, 1)
        multi_ops: List[np.ndarray] = [single_ops[0]]
        for op in single_ops[1:]:
            multi_ops.append(op)
        # For simplicity, use the n-qubit depolarising channel:
        # K0 = sqrt(1-p) I, K_i = sqrt(p/(4^n - 1)) P_i for all Pauli strings except I
        identity = np.eye(dim, dtype=_COMPLEX_DTYPE)
        k0 = math.sqrt(max(0.0, 1.0 - p)) * identity
        num_paulis = (1 << (2 * num_qubits)) - 1
        coeff = math.sqrt(max(0.0, p / num_paulis))
        paulis = [X, np.array([[0, -1j], [1j, 0]], dtype=_COMPLEX_DTYPE),
                  np.array([[1, 0], [0, -1]], dtype=_COMPLEX_DTYPE)]
        ops = [k0]
        for i in range(1, 1 << (2 * num_qubits)):
            # Decompose i into Pauli string
            mat = np.eye(1, dtype=_COMPLEX_DTYPE)
            temp = i
            for _ in range(num_qubits):
                mat = np.kron(paulis[temp & 3], mat)
                temp >>= 2
            ops.append(coeff * mat)
        return ops


def _amplitude_damping_kraus(
    params: Tuple[float, ...],
    num_qubits: int,
) -> List[np.ndarray]:
    """Amplitude damping channel (single qubit).

    Parameters
    ----------
    params : tuple of float
        ``[gamma]`` — damping probability.

    Kraus operators::

        K₀ = [[1, 0], [0, √(1 − γ)]]
        K₁ = [[0, √γ], [0, 0]]
    """
    gamma = params[0] if params else 0.0
    gamma = max(0.0, min(1.0, gamma))

    K0 = np.array(
        [[1, 0], [0, math.sqrt(1.0 - gamma)]],
        dtype=_COMPLEX_DTYPE,
    )
    K1 = np.array(
        [[0, math.sqrt(gamma)], [0, 0]],
        dtype=_COMPLEX_DTYPE,
    )

    if num_qubits == 1:
        return [K0, K1]
    else:
        # Apply to the first qubit, identity on the rest
        I_rest = np.eye(1 << (num_qubits - 1), dtype=_COMPLEX_DTYPE)
        ops: List[np.ndarray] = []
        for K in [K0, K1]:
            ops.append(np.kron(K, I_rest))
        return ops


def _phase_damping_kraus(
    params: Tuple[float, ...],
    num_qubits: int,
) -> List[np.ndarray]:
    """Phase damping (dephasing) channel.

    Parameters
    ----------
    params : tuple of float
        ``[lambda_]`` — dephasing probability.

    Kraus operators::

        K₀ = √(1 − λ/2) I
        K₁ = √(λ/2) Z
    """
    lam = params[0] if params else 0.0
    lam = max(0.0, min(1.0, lam))

    coeff0 = math.sqrt(1.0 - lam / 2.0)
    coeff1 = math.sqrt(lam / 2.0)

    I2 = np.eye(2, dtype=_COMPLEX_DTYPE)
    Z = np.array([[1, 0], [0, -1]], dtype=_COMPLEX_DTYPE)

    if num_qubits == 1:
        return [coeff0 * I2, coeff1 * Z]
    else:
        I_rest = np.eye(1 << (num_qubits - 1), dtype=_COMPLEX_DTYPE)
        return [
            coeff0 * np.kron(I2, I_rest),
            coeff1 * np.kron(Z, I_rest),
        ]


def _bit_flip_kraus(
    params: Tuple[float, ...],
    num_qubits: int,
) -> List[np.ndarray]:
    """Bit-flip channel.

    Parameters
    ----------
    params : tuple of float
        ``[p]`` — flip probability.

    Kraus operators::

        K₀ = √(1 − p) I
        K₁ = √p X
    """
    p = params[0] if params else 0.0
    p = max(0.0, min(1.0, p))

    I2 = np.eye(2, dtype=_COMPLEX_DTYPE)
    X = np.array([[0, 1], [1, 0]], dtype=_COMPLEX_DTYPE)

    if num_qubits == 1:
        return [math.sqrt(1.0 - p) * I2, math.sqrt(p) * X]
    else:
        I_rest = np.eye(1 << (num_qubits - 1), dtype=_COMPLEX_DTYPE)
        return [
            math.sqrt(1.0 - p) * np.kron(I2, I_rest),
            math.sqrt(p) * np.kron(X, I_rest),
        ]


def _phase_flip_kraus(
    params: Tuple[float, ...],
    num_qubits: int,
) -> List[np.ndarray]:
    """Phase-flip channel.

    Parameters
    ----------
    params : tuple of float
        ``[p]`` — flip probability.

    Kraus operators::

        K₀ = √(1 − p) I
        K₁ = √p Z
    """
    p = params[0] if params else 0.0
    p = max(0.0, min(1.0, p))

    I2 = np.eye(2, dtype=_COMPLEX_DTYPE)
    Z = np.array([[1, 0], [0, -1]], dtype=_COMPLEX_DTYPE)

    if num_qubits == 1:
        return [math.sqrt(1.0 - p) * I2, math.sqrt(p) * Z]
    else:
        I_rest = np.eye(1 << (num_qubits - 1), dtype=_COMPLEX_DTYPE)
        return [
            math.sqrt(1.0 - p) * np.kron(I2, I_rest),
            math.sqrt(p) * np.kron(Z, I_rest),
        ]


# Registry of noise channels
_NOISE_CHANNEL_REGISTRY: Dict[
    str,
    Any,
] = {
    "depolarizing": _depolarizing_kraus,
    "amplitude_damping": _amplitude_damping_kraus,
    "phase_damping": _phase_damping_kraus,
    "dephasing": _phase_damping_kraus,
    "bit_flip": _bit_flip_kraus,
    "phase_flip": _phase_flip_kraus,
}


# ---------------------------------------------------------------------------
# DensityMatrixBackend
# ---------------------------------------------------------------------------

class DensityMatrixBackend:
    """Full density-matrix simulation backend.

    Maintains a ``(2ⁿ, 2ⁿ)`` density matrix ``ρ`` and provides operations
    for unitary evolution, Kraus-map application, measurement, partial
    trace, and noise simulation.

    Parameters
    ----------
    precision : str, optional
        ``'double'`` (default) or ``'single'``.
    seed : int or None, optional
        Random seed for measurement sampling.

    Examples
    --------
    >>> backend = DensityMatrixBackend()
    >>> rho = backend.zero_state(2)
    >>> rho.shape
    (4, 4)
    """

    def __init__(
        self,
        precision: str = "double",
        seed: Optional[int] = None,
    ) -> None:
        if precision == "single":
            self._dtype = np.complex64
            self._float_dtype = np.float32
        else:
            self._dtype = _COMPLEX_DTYPE
            self._float_dtype = _FLOAT_DTYPE

        self._rng = np.random.default_rng(seed)
        self._gate_cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = {}

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def zero_state(self, num_qubits: int) -> np.ndarray:
        """Return the pure ``|00…0⟩⟨00…0|`` density matrix.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        numpy.ndarray
            Shape ``(2**n, 2**n)``.
        """
        dim = 1 << num_qubits
        rho = np.zeros((dim, dim), dtype=self._dtype)
        rho[0, 0] = 1.0
        return rho

    def maximally_mixed_state(self, num_qubits: int) -> np.ndarray:
        """Return the maximally mixed state ``I / 2ⁿ``.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        numpy.ndarray
        """
        dim = 1 << num_qubits
        return np.eye(dim, dtype=self._dtype) / dim

    def from_statevector(self, sv: Union[Statevector, np.ndarray]) -> np.ndarray:
        """Convert a statevector to a density matrix: ``ρ = |ψ⟩⟨ψ|``.

        Parameters
        ----------
        sv : Statevector or numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(sv, Statevector):
            data = sv.data.astype(self._dtype)
        else:
            data = np.asarray(sv, dtype=self._dtype)
        return np.outer(data, np.conj(data))

    def from_density_matrix(
        self,
        dm: Union[CoreDensityMatrix, np.ndarray],
    ) -> np.ndarray:
        """Import a :class:`DensityMatrix` (or raw array) into this
        backend's representation.

        Parameters
        ----------
        dm : DensityMatrix or numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(dm, CoreDensityMatrix):
            return dm.data.astype(self._dtype, copy=True)
        return np.asarray(dm, dtype=self._dtype, copy=True)

    # ------------------------------------------------------------------
    # Gate matrix cache
    # ------------------------------------------------------------------

    def _get_gate_matrix(
        self,
        gate: Gate,
        params: Tuple[float, ...] = (),
    ) -> np.ndarray:
        """Retrieve (and cache) the gate matrix."""
        cache_key = (gate.name, params)
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]

        if params:
            mat = gate.to_matrix(*params)
        else:
            mat = gate.matrix
        mat = mat.astype(self._dtype, copy=False)
        self._gate_cache[cache_key] = mat
        return mat

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_gate(
        self,
        rho: np.ndarray,
        gate: Gate,
        qubits: Sequence[int],
        params: Tuple[float, ...] = (),
    ) -> np.ndarray:
        r"""Apply a unitary gate: ``ρ' = U ρ U†``.

        Parameters
        ----------
        rho : numpy.ndarray
            Density matrix of shape ``(2ⁿ, 2ⁿ)``. Modified **in-place**.
        gate : Gate
        qubits : sequence of int
        params : tuple of float

        Returns
        -------
        numpy.ndarray
        """
        n = int(round(math.log2(rho.shape[0])))
        gate_matrix = self._get_gate_matrix(gate, params)
        full_U = self._embed_operator(gate_matrix, qubits, n)
        rho[:] = full_U @ rho @ full_U.conj().T
        return rho

    def apply_gate_full(
        self,
        rho: np.ndarray,
        gate_matrix: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        r"""Apply an arbitrary unitary: ``ρ' = U ρ U†``.

        Parameters
        ----------
        rho : numpy.ndarray
        gate_matrix : numpy.ndarray
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        numpy.ndarray
        """
        full_U = self._embed_operator(gate_matrix, qubits, num_qubits)
        rho[:] = full_U @ rho @ full_U.conj().T
        return rho

    # ------------------------------------------------------------------
    # Kraus operator application
    # ------------------------------------------------------------------

    def apply_kraus(
        self,
        rho: np.ndarray,
        kraus_ops: Sequence[np.ndarray],
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        r"""Apply a general quantum operation via Kraus operators.

        .. math:: ρ' = \sum_k E_k ρ E_k^†

        Parameters
        ----------
        rho : numpy.ndarray
            Density matrix of shape ``(2ⁿ, 2ⁿ)``. Modified in-place.
        kraus_ops : sequence of numpy.ndarray
            Each of shape ``(2**k, 2**k)`` where ``k = len(qubits)``.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        numpy.ndarray
        """
        result = np.zeros_like(rho)
        for K in kraus_ops:
            K = np.asarray(K, dtype=self._dtype)
            K_full = self._embed_operator(K, qubits, num_qubits)
            result += K_full @ rho @ K_full.conj().T

        # Renormalise trace
        tr = np.trace(result)
        if tr > _TOLERANCE:
            result /= tr
        rho[:] = result
        return rho

    # ------------------------------------------------------------------
    # Noise application
    # ------------------------------------------------------------------

    def apply_noise(
        self,
        rho: np.ndarray,
        noise_channel: NoiseChannel,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Apply a noise channel to the specified qubits.

        Parameters
        ----------
        rho : numpy.ndarray
            Density matrix of shape ``(2ⁿ, 2ⁿ)``. Modified in-place.
        noise_channel : NoiseChannel
            The noise channel to apply.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        numpy.ndarray
        """
        k = len(qubits)
        kraus_ops = noise_channel.kraus_operators(k)
        return self.apply_kraus(rho, kraus_ops, qubits, num_qubits)

    def apply_depolarizing(
        self,
        rho: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
        p: float = 0.01,
    ) -> np.ndarray:
        """Apply the depolarising channel.

        Parameters
        ----------
        rho : numpy.ndarray
        qubits : sequence of int
        num_qubits : int
        p : float
            Error probability.

        Returns
        -------
        numpy.ndarray
        """
        channel = NoiseChannel("depolarizing", (p,))
        return self.apply_noise(rho, channel, qubits, num_qubits)

    def apply_amplitude_damping(
        self,
        rho: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
        gamma: float = 0.01,
    ) -> np.ndarray:
        """Apply the amplitude damping channel.

        Parameters
        ----------
        rho : numpy.ndarray
        qubits : sequence of int
        num_qubits : int
        gamma : float
            Damping probability.

        Returns
        -------
        numpy.ndarray
        """
        channel = NoiseChannel("amplitude_damping", (gamma,))
        return self.apply_noise(rho, channel, qubits, num_qubits)

    def apply_phase_damping(
        self,
        rho: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
        lam: float = 0.01,
    ) -> np.ndarray:
        """Apply the phase damping (dephasing) channel.

        Parameters
        ----------
        rho : numpy.ndarray
        qubits : sequence of int
        num_qubits : int
        lam : float
            Dephasing probability.

        Returns
        -------
        numpy.ndarray
        """
        channel = NoiseChannel("phase_damping", (lam,))
        return self.apply_noise(rho, channel, qubits, num_qubits)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def apply_measurement(
        self,
        rho: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Perform a POVM measurement on the specified qubits.

        The state is collapsed according to the Born rule.

        Parameters
        ----------
        rho : numpy.ndarray
            Density matrix of shape ``(2ⁿ, 2ⁿ)``. Modified in-place.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        outcome : int
            Measurement outcome as an integer.
        new_rho : numpy.ndarray
            Post-measurement density matrix.
        probabilities : numpy.ndarray
            Marginal probabilities, shape ``(2**k,)``.
        """
        qubits = list(qubits)
        n = num_qubits
        k = len(qubits)
        dim = rho.shape[0]

        # Compute marginal probabilities via partial trace over complement
        # Use the computational basis projectors
        probs = np.zeros(1 << k, dtype=self._float_dtype)
        for outcome in range(1 << k):
            # Build projector for this outcome on the target qubits
            proj = self._build_projector(outcome, qubits, n)
            prob = np.real(np.trace(proj @ rho))
            probs[outcome] = max(0.0, prob)

        # Normalise probabilities
        total = probs.sum()
        if total > _TOLERANCE:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

        # Sample outcome
        outcome = int(self._rng.choice(len(probs), p=probs))

        # Collapse: ρ' = P_outcome ρ P_outcome / p(outcome)
        proj = self._build_projector(outcome, qubits, n)
        rho[:] = proj @ rho @ proj
        p_outcome = probs[outcome]
        if p_outcome > _TOLERANCE:
            rho /= p_outcome
        else:
            # Fallback: reset to zero state
            rho[:] = self.zero_state(n)

        return outcome, rho, probs

    def _build_projector(
        self,
        outcome: int,
        qubits: List[int],
        n: int,
    ) -> np.ndarray:
        """Build the projector ``|outcome⟩⟨outcome|`` on the target
        qubits, tensored with identity on the rest.

        Parameters
        ----------
        outcome : int
        qubits : list of int
        n : int

        Returns
        -------
        numpy.ndarray
            Shape ``(2**n, 2**n)``.
        """
        k = len(qubits)
        dim = 1 << n

        # Build computational basis state on target qubits
        target_state = np.zeros(1 << k, dtype=self._dtype)
        target_state[outcome] = 1.0

        # Embed: P = |outcome⟩⟨outcome|_targets ⊗ I_rest
        target_proj = np.outer(target_state, target_state)
        P_full = self._embed_operator(target_proj, qubits, n)
        return P_full

    # ------------------------------------------------------------------
    # Partial trace
    # ------------------------------------------------------------------

    def partial_trace(
        self,
        rho: np.ndarray,
        keep_qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Trace out qubits not in *keep_qubits*.

        Uses efficient tensor reshaping and axis summation.

        Parameters
        ----------
        rho : numpy.ndarray
            Density matrix of shape ``(2ⁿ, 2ⁿ)``.
        keep_qubits : sequence of int
            Qubit indices to retain.
        num_qubits : int

        Returns
        -------
        numpy.ndarray
            Reduced density matrix of shape ``(2**k, 2**k)`` where
            ``k = len(keep_qubits)``.
        """
        n = num_qubits
        keep = sorted(set(keep_qubits))
        trace_out = sorted(set(range(n)) - set(keep))

        if not trace_out:
            return rho.copy()
        if not keep:
            dim = 1 << n
            return np.array([[np.trace(rho)]], dtype=self._dtype)

        k = len(keep)
        reduced = _partial_trace_impl(rho, n, keep, trace_out)
        return reduced

    # ------------------------------------------------------------------
    # Expectation values
    # ------------------------------------------------------------------

    def expectation_value(
        self,
        rho: np.ndarray,
        observable: np.ndarray,
    ) -> float:
        r"""Compute ``Tr(O ρ)`` for a Hermitian observable.

        Parameters
        ----------
        rho : numpy.ndarray
        observable : numpy.ndarray

        Returns
        -------
        float
        """
        result = np.trace(observable @ rho)
        return float(np.real(result))

    def expectation_value_on_qubits(
        self,
        rho: np.ndarray,
        observable: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> float:
        r"""Compute ``Tr((O ⊗ I_rest) ρ)``.

        Parameters
        ----------
        rho : numpy.ndarray
        observable : numpy.ndarray
            Shape ``(2**k, 2**k)``.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        float
        """
        full_obs = self._embed_operator(observable, qubits, num_qubits)
        return self.expectation_value(rho, full_obs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def purity(self, rho: np.ndarray) -> float:
        r"""Compute ``Tr(ρ²)``.

        Returns 1.0 for a pure state, < 1.0 for a mixed state.

        Parameters
        ----------
        rho : numpy.ndarray

        Returns
        -------
        float
        """
        return float(np.real(np.trace(rho @ rho)))

    def von_neumann_entropy(self, rho: np.ndarray) -> float:
        r"""Compute ``S(ρ) = -Tr(ρ log₂ ρ)``.

        Parameters
        ----------
        rho : numpy.ndarray

        Returns
        -------
        float
        """
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > _TOLERANCE]
        if len(eigenvalues) == 0:
            return 0.0
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def is_pure(self, rho: np.ndarray, tol: float = 1e-8) -> bool:
        """Check whether the state is pure (purity ≈ 1).

        Parameters
        ----------
        rho : numpy.ndarray
        tol : float

        Returns
        -------
        bool
        """
        return abs(self.purity(rho) - 1.0) < tol

    def fidelity(
        self,
        rho1: np.ndarray,
        rho2: np.ndarray,
    ) -> float:
        r"""Compute the Uhlmann fidelity between two density matrices.

        .. math:: F(ρ, σ) = (Tr √{√ρ σ √ρ})²

        Parameters
        ----------
        rho1 : numpy.ndarray
        rho2 : numpy.ndarray

        Returns
        -------
        float
        """
        sqrt_rho = _matrix_sqrt(rho1)
        product = sqrt_rho @ rho2 @ sqrt_rho
        sqrt_product = _matrix_sqrt(product)
        fid = float(np.real(np.trace(sqrt_product) ** 2))
        return min(1.0, max(0.0, fid))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        rho: np.ndarray,
        shots: int,
        num_qubits: int,
    ) -> Dict[str, int]:
        """Sample measurement outcomes from a density matrix.

        Uses the Born rule with the diagonal elements (probabilities).

        Parameters
        ----------
        rho : numpy.ndarray
            Shape ``(2ⁿ, 2ⁿ)``.
        shots : int
        num_qubits : int

        Returns
        -------
        dict
            Mapping from bit-string to count.
        """
        probs = np.real(np.diag(rho))
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > _TOLERANCE:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

        outcomes = self._rng.choice(len(probs), size=shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            bits = format(int(o), f"0{num_qubits}b")
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    def probabilities(self, rho: np.ndarray) -> np.ndarray:
        """Return the diagonal of the density matrix (measurement
        probabilities for each computational basis state).

        Parameters
        ----------
        rho : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Shape ``(2ⁿ,)``.
        """
        return np.real(np.diag(rho))

    # ------------------------------------------------------------------
    # Circuit execution
    # ------------------------------------------------------------------

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[np.ndarray] = None,
        noise_model: Optional[Any] = None,
    ) -> np.ndarray:
        """Execute a quantum circuit and return the final density matrix.

        Parameters
        ----------
        circuit : QuantumCircuit
        initial_state : numpy.ndarray, optional
            Initial density matrix. Defaults to ``|0⟩⟨0|``.
        noise_model : object, optional
            An object that provides ``after_gate(rho, gate, qubits)``.
            If provided, noise is applied after every gate.

        Returns
        -------
        numpy.ndarray
            Final density matrix.
        """
        n = circuit.num_qubits
        if initial_state is not None:
            rho = initial_state.copy().astype(self._dtype)
        else:
            rho = self.zero_state(n)

        for op in circuit.data:
            if isinstance(op, Barrier):
                continue
            if isinstance(op, Reset):
                rho = self._apply_reset(rho, op.qubits, n)
                continue
            if isinstance(op, Operation):
                if isinstance(op.gate, Measurement):
                    rho = self._apply_measurement_op(rho, op.qubits, n)
                    continue
                self.apply_gate(rho, op.gate, op.qubits, op.params)
                # Apply noise model if provided
                if noise_model is not None and hasattr(noise_model, "after_gate"):
                    rho = noise_model.after_gate(rho, op.gate, op.qubits, n)
                continue

        return rho

    def _apply_reset(
        self,
        rho: np.ndarray,
        qubits: Tuple[int, ...],
        n: int,
    ) -> np.ndarray:
        """Reset qubits to |0⟩.

        Implements a projective measurement discard followed by
        preparation of |0⟩.

        Parameters
        ----------
        rho : numpy.ndarray
        qubits : tuple of int
        n : int

        Returns
        -------
        numpy.ndarray
        """
        dim = rho.shape[0]
        for q in qubits:
            # Projector for |0⟩ on qubit q
            proj = self._build_projector(0, [q], n)
            rho = proj @ rho @ proj
            # Renormalise
            tr = np.trace(rho)
            if tr > _TOLERANCE:
                rho /= tr
        return rho

    def _apply_measurement_op(
        self,
        rho: np.ndarray,
        qubits: Tuple[int, ...],
        n: int,
    ) -> np.ndarray:
        """Apply a measurement operation and collapse.

        Parameters
        ----------
        rho : numpy.ndarray
        qubits : tuple of int
        n : int

        Returns
        -------
        numpy.ndarray
        """
        _, rho, _ = self.apply_measurement(rho, list(qubits), n)
        return rho

    # ------------------------------------------------------------------
    # Operator embedding
    # ------------------------------------------------------------------

    def _embed_operator(
        self,
        operator: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Embed an operator on target qubits into the full Hilbert space.

        Parameters
        ----------
        operator : numpy.ndarray
            Shape ``(2**k, 2**k)``.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        numpy.ndarray
            Shape ``(2**n, 2**n)``.
        """
        n = num_qubits
        k = len(qubits)
        dim = 1 << n

        if k == n:
            return operator
        if k == 0:
            return np.eye(dim, dtype=self._dtype)

        perm = list(qubits) + [q for q in range(n) if q not in qubits]
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i

        gate_on_front = np.kron(
            operator, np.eye(1 << (n - k), dtype=self._dtype)
        )
        P = _permutation_matrix(perm, n)
        P_inv = _permutation_matrix(inv_perm, n)
        return P_inv @ gate_on_front @ P

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the gate-matrix cache."""
        self._gate_cache.clear()

    def to_core_density_matrix(
        self,
        rho: np.ndarray,
        num_qubits: int,
    ) -> CoreDensityMatrix:
        """Wrap the raw array in a core :class:`DensityMatrix`.

        Parameters
        ----------
        rho : numpy.ndarray
        num_qubits : int

        Returns
        -------
        DensityMatrix
        """
        return CoreDensityMatrix(rho, num_qubits=num_qubits, copy=False)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _partial_trace_impl(
    rho: np.ndarray,
    n: int,
    keep: List[int],
    trace_out: List[int],
) -> np.ndarray:
    """Efficient partial trace via tensor reshaping.

    Parameters
    ----------
    rho : numpy.ndarray
        Shape ``(2ⁿ, 2ⁿ)``.
    n : int
    keep : list of int
    trace_out : list of int

    Returns
    -------
    numpy.ndarray
    """
    # Reshape rho to (2, 2, ..., 2) with 2n indices
    # Row indices: i₀, i₁, ..., i_{n-1}
    # Col indices: j₀, j₁, ..., j_{n-1}
    tensor = rho.reshape([2] * (2 * n))

    # For each traced-out qubit, set row and col index equal and sum
    for q in trace_out:
        # Row index for qubit q is at position q
        # Col index for qubit q is at position n + q
        tensor = np.trace(tensor, axis1=q, axis2=n + q)
        # After tracing out, tensor loses one dimension
        # We need to adjust indices for subsequent qubits
        n -= 1
        # Adjust keep and remaining trace_out indices
        keep = [k if k < q else k - 1 for k in keep if k != q]
        trace_out = [t if t < q else t - 1 for t in trace_out if t != q]

    # Now tensor has shape (2, 2, ..., 2) with 2k indices
    # Reshape to (2**k, 2**k)
    k = len(keep)
    return tensor.reshape((1 << k, 1 << k))


def _matrix_sqrt(m: np.ndarray) -> np.ndarray:
    """Compute the principal square root of a positive semi-definite
    matrix via eigendecomposition.

    Parameters
    ----------
    m : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.conj().T


def _permutation_matrix(perm: List[int], n: int) -> np.ndarray:
    """Build a unitary permutation matrix for qubit reordering.

    Parameters
    ----------
    perm : list of int
        ``perm[i]`` = which original qubit goes to position ``i``.
    n : int

    Returns
    -------
    numpy.ndarray
    """
    dim = 1 << n
    P = np.zeros((dim, dim), dtype=_COMPLEX_DTYPE)
    for i in range(dim):
        original = 0
        for pos in range(n):
            original_qubit = perm[pos]
            bit = (i >> (n - 1 - pos)) & 1
            original |= bit << (n - 1 - original_qubit)
        P[original, i] = 1.0
    return P
