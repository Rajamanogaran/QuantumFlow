"""
Statevector Simulation Backend
================================

High-performance pure-state quantum circuit simulator that operates on the
full statevector ``|ψ⟩`` of shape ``(2ⁿ,)``.

The :class:`StatevectorBackend` provides:

* Gate application via efficient ``numpy.einsum`` / tensor contraction.
* Projective measurement with state collapse.
* Expectation-value computation ``⟨ψ|O|ψ⟩``.
* Sampling of measurement outcomes (Born rule).
* Batch simulation: apply the same circuit to many initial states in
  parallel by stacking statevectors along the leading axis.
* Numerical-gradient computation for variational algorithms (VQA).
* Optional GPU acceleration when the input arrays already reside on a
  GPU device (e.g. ``cupy``-compatible).

Numerical stability
-------------------
After every gate application the statevector is normalised so that
``‖ψ‖₂ = 1`` up to floating-point precision, preventing the accumulation
of rounding errors in deep circuits.

Typical usage::

    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.simulation.statevector import StatevectorBackend

    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)

    backend = StatevectorBackend()
    state = backend.run_circuit(qc)
    probs = backend.probabilities(state)

    # Batch simulation
    states = backend.run_circuit_batch(qc, batch_size=16)
"""

from __future__ import annotations

import math
import time
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import Gate, Measurement
from quantumflow.core.operation import Barrier, Operation, Reset
from quantumflow.core.state import Statevector

__all__ = [
    "StatevectorBackend",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEX_DTYPE = np.complex128
_FLOAT_DTYPE = np.float64
_TOLERANCE = 1e-12
_EPS = 1e-15

# Try to detect cupy for optional GPU support
try:
    import cupy as _cp  # type: ignore[import-untyped]

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# Helper: resolve the array library (numpy / cupy)
# ---------------------------------------------------------------------------

def _get_xp(arr: np.ndarray) -> Any:
    """Return the array module (numpy or cupy) for *arr*."""
    if _HAS_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return _cp
    return np


def _asarray(
    data: Union[np.ndarray, Sequence, Sequence[Sequence]],
    dtype: Any = _COMPLEX_DTYPE,
    copy: bool = False,
) -> np.ndarray:
    """Convert *data* to an ndarray of the requested dtype."""
    if _HAS_CUPY and hasattr(data, "__cuda_array_interface__"):
        xp = _cp
    else:
        xp = np
    return xp.asarray(data, dtype=dtype)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """In-place normalise *arr* and return it."""
    xp = _get_xp(arr)
    norm = xp.linalg.norm(arr)
    if norm > _TOLERANCE:
        arr /= norm
    return arr


# ---------------------------------------------------------------------------
# StatevectorBackend
# ---------------------------------------------------------------------------

class StatevectorBackend:
    """Full statevector simulation backend.

    Maintains a complex vector ``|ψ⟩`` of shape ``(2ⁿ,)`` and provides
    all operations needed for noiseless quantum circuit simulation.

    Parameters
    ----------
    precision : str, optional
        ``'double'`` (complex128, default) or ``'single'`` (complex64).
    seed : int or None, optional
        Random seed for reproducible sampling.

    Examples
    --------
    >>> backend = StatevectorBackend()
    >>> state = backend.zero_state(3)
    >>> state.shape
    (8,)
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
        """Return the ``|00…0⟩`` statevector.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        numpy.ndarray
            Shape ``(2**num_qubits,)``, dtype ``complex128``.
        """
        dim = 1 << num_qubits
        state = np.zeros(dim, dtype=self._dtype)
        state[0] = 1.0
        return state

    def from_statevector(self, sv: Union[Statevector, np.ndarray]) -> np.ndarray:
        """Convert a :class:`Statevector` (or raw array) to the internal
        representation used by this backend.

        Parameters
        ----------
        sv : Statevector or numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(sv, Statevector):
            return _asarray(sv.data, dtype=self._dtype, copy=True)
        return _asarray(sv, dtype=self._dtype, copy=True)

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def _get_gate_matrix(
        self,
        gate: Gate,
        params: Tuple[float, ...] = (),
    ) -> np.ndarray:
        """Retrieve (and cache) the matrix for *gate*.

        For parameterised gates the matrix is recomputed when *params*
        differs from the cached entry.
        """
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

    def apply_gate(
        self,
        state: np.ndarray,
        gate: Gate,
        qubits: Sequence[int],
        params: Tuple[float, ...] = (),
    ) -> np.ndarray:
        """Apply a gate to specific qubits of the statevector.

        Uses an efficient tensor-contraction strategy:

        1. The statevector is reshaped to ``(2, 2, …, 2)`` (one axis per
           qubit).
        2. The gate matrix is reshaped to ``(2, 2, 2, 2, …)``
           (two input axes + two output axes per gate qubit).
        3. ``numpy.einsum`` contracts the appropriate axes, and the result
           is reshaped back to a flat vector.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.  Modified **in-place**.
        gate : Gate
            Quantum gate with ``gate.num_qubits == len(qubits)``.
        qubits : sequence of int
            Target qubit indices (0-based, most-significant-first in the
            amplitude ordering convention used by QuantumFlow).
        params : tuple of float
            Parameter values for parameterised gates.

        Returns
        -------
        numpy.ndarray
            The updated statevector (same object as *state*).
        """
        n = int(round(math.log2(state.shape[0])))
        k = len(qubits)

        if k == 0:
            return state

        gate_matrix = self._get_gate_matrix(gate, params)

        # Handle full-system gate (optimisation)
        if k == n:
            state[:] = gate_matrix @ state
            _normalize(state)
            return state

        # --- Tensor-contraction approach ----------------------------------
        # Reshape state to (2, 2, ..., 2)
        shape = [2] * n
        psi = state.reshape(shape)

        # Reshape gate to (2, 2, ..., 2) with 2k indices total
        # The first k indices are output (row), the last k are input (col)
        gate_tensor = gate_matrix.reshape([2] * (2 * k))

        # Build einsum subscripts
        # State indices: a0 a1 a2 ... a_{n-1}
        # We replace the k target indices with output+input indices
        state_indices = list("abcdefghijklmnopqrstuvwxyz"[:n])
        output_indices = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k])
        input_indices = list(state_indices[q] for q in qubits)

        # New state indices: replace target qubit indices with output indices
        new_state_indices = list(state_indices)
        for i, q in enumerate(qubits):
            new_state_indices[q] = output_indices[i]

        # Einsum string: gate subscripts + state subscripts -> new state subscripts
        gate_subs = "".join(output_indices + input_indices)
        state_subs = "".join(state_indices)
        result_subs = "".join(new_state_indices)

        einsum_str = f"{gate_subs},{state_subs}->{result_subs}"

        # Rearrange axes of gate_tensor so they correspond to the qubit ordering
        # The gate's first k indices are output, last k are input
        # But the qubits may not be in order 0,1,...,k-1
        # We need to reorder gate_tensor axes to match qubit ordering

        # Sort qubits to get the canonical order, then permute gate_tensor accordingly
        sorted_positions = sorted(range(k), key=lambda i: qubits[i])

        # Permute gate_tensor: output axes by sorted_positions, input axes by sorted_positions
        perm = sorted_positions + [k + p for p in sorted_positions]
        gate_tensor_reordered = np.transpose(gate_tensor, perm)

        psi = np.einsum(einsum_str, gate_tensor_reordered, psi, optimize=True)

        state[:] = psi.reshape(-1)
        _normalize(state)
        return state

    def apply_gate_full(
        self,
        state: np.ndarray,
        gate_matrix: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Apply an arbitrary gate matrix to specific qubits.

        This is a convenience wrapper around :meth:`apply_gate` that
        accepts a raw numpy matrix instead of a :class:`Gate` object.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.
        gate_matrix : numpy.ndarray
            Unitary matrix of shape ``(2**k, 2**k)``.
        qubits : sequence of int
            Target qubit indices.
        num_qubits : int
            Total number of qubits.

        Returns
        -------
        numpy.ndarray
        """
        n = num_qubits
        k = len(qubits)

        if k == n:
            state[:] = gate_matrix @ state
            _normalize(state)
            return state

        shape = [2] * n
        psi = state.reshape(shape)

        gate_tensor = gate_matrix.reshape([2] * (2 * k))

        state_indices = list("abcdefghijklmnopqrstuvwxyz"[:n])
        output_indices = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k])
        input_indices = list(state_indices[q] for q in qubits)

        new_state_indices = list(state_indices)
        for i, q in enumerate(qubits):
            new_state_indices[q] = output_indices[i]

        gate_subs = "".join(output_indices + input_indices)
        state_subs = "".join(state_indices)
        result_subs = "".join(new_state_indices)

        einsum_str = f"{gate_subs},{state_subs}->{result_subs}"

        sorted_positions = sorted(range(k), key=lambda i: qubits[i])
        perm = sorted_positions + [k + p for p in sorted_positions]
        gate_tensor_reordered = np.transpose(gate_tensor, perm)

        psi = np.einsum(einsum_str, gate_tensor_reordered, psi, optimize=True)
        state[:] = psi.reshape(-1)
        _normalize(state)
        return state

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def apply_measurement(
        self,
        state: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Perform a projective measurement on the given qubits.

        The state is **collapsed in-place** to the post-measurement state.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``. Modified in-place.
        qubits : sequence of int
            Qubits to measure.
        num_qubits : int
            Total number of qubits.

        Returns
        -------
        outcome : int
            Measurement outcome (integer encoding).
        new_state : numpy.ndarray
            Collapsed statevector (normalised).
        probabilities : numpy.ndarray
            Marginal probabilities for the measured sub-system, shape
            ``(2**len(qubits),)``.
        """
        qubits = list(qubits)
        n = num_qubits
        dim = state.shape[0]
        k = len(qubits)

        # Compute marginal probabilities
        probs = np.abs(state) ** 2
        # Reshape to tensor form and marginalise
        tensor = probs.reshape([2] * n)
        keep_axes = tuple(qubits)
        # Sum over non-measured axes
        all_axes = set(range(n))
        trace_axes = tuple(sorted(all_axes - set(keep_axes)))
        marginal = tensor
        for ax in trace_axes:
            marginal = marginal.sum(axis=ax)
        marginal_flat = marginal.reshape(-1)
        marginal_flat = np.real(marginal_flat)

        # Sample outcome
        total = marginal_flat.sum()
        if total > _TOLERANCE:
            marginal_flat /= total
        outcome = int(self._rng.choice(len(marginal_flat), p=marginal_flat))

        # Collapse state
        # Build mask for the outcome
        outcome_bits = [
            (outcome >> (k - 1 - i)) & 1 for i in range(k)
        ]
        mask = np.ones(dim, dtype=bool)
        for qi, bit in zip(qubits, outcome_bits):
            qmask = np.array(
                [((idx >> (n - 1 - qi)) & 1) == bit for idx in range(dim)],
                dtype=bool,
            )
            mask &= qmask

        state[~mask] = 0.0
        _normalize(state)

        return outcome, state, marginal_flat

    # ------------------------------------------------------------------
    # Expectation values
    # ------------------------------------------------------------------

    def expectation_value(
        self,
        state: np.ndarray,
        observable: np.ndarray,
    ) -> float:
        r"""Compute ``⟨ψ|O|ψ⟩`` for a Hermitian observable *O*.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.
        observable : numpy.ndarray
            Hermitian matrix of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        float
            Real expectation value.
        """
        O = _asarray(observable, dtype=self._dtype)
        result = np.vdot(state, O @ state)
        return float(np.real(result))

    def expectation_value_on_qubits(
        self,
        state: np.ndarray,
        observable: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> float:
        r"""Compute ``⟨ψ|O_qubits ⊗ I_rest|ψ⟩``.

        Embeds the observable on the specified qubits with identity on
        the rest of the system.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.
        observable : numpy.ndarray
            Observable matrix of shape ``(2**k, 2**k)`` acting on the
            target qubits.
        qubits : sequence of int
            Target qubit indices.
        num_qubits : int
            Total number of qubits.

        Returns
        -------
        float
        """
        full_obs = self._embed_operator(observable, qubits, num_qubits)
        return self.expectation_value(state, full_obs)

    def _embed_operator(
        self,
        operator: np.ndarray,
        qubits: Sequence[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Embed *operator* into the full Hilbert space.

        Places *operator* on *qubits* and identity on all other qubits
        via the permutation-based Kronecker product strategy.

        Parameters
        ----------
        operator : numpy.ndarray
            Matrix of shape ``(2**k, 2**k)``.
        qubits : sequence of int
            Target qubit indices.
        num_qubits : int
            Total number of qubits ``n``.

        Returns
        -------
        numpy.ndarray
            Full matrix of shape ``(2**n, 2**n)``.
        """
        n = num_qubits
        k = len(qubits)
        dim = 1 << n

        if k == n:
            return operator
        if k == 0:
            return np.eye(dim, dtype=self._dtype)

        # Permutation-based embedding
        perm = list(qubits) + [q for q in range(n) if q not in qubits]
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i

        gate_on_front = np.kron(
            operator, np.eye(1 << (n - k), dtype=self._dtype)
        )
        P = self._permutation_matrix(perm, n)
        P_inv = self._permutation_matrix(inv_perm, n)
        return P_inv @ gate_on_front @ P

    @staticmethod
    def _permutation_matrix(perm: List[int], n: int) -> np.ndarray:
        """Build a unitary permutation matrix for qubit reordering.

        Parameters
        ----------
        perm : list of int
            ``perm[i]`` = which original qubit goes to position ``i``.
        n : int
            Number of qubits.

        Returns
        -------
        numpy.ndarray
            Shape ``(2**n, 2**n)``.
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        state: np.ndarray,
        shots: int,
        num_qubits: int,
    ) -> Dict[str, int]:
        """Sample measurement outcomes from a statevector.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.
        shots : int
            Number of repetitions.
        num_qubits : int
            Number of qubits.

        Returns
        -------
        dict
            Mapping from bit-string (e.g. ``'010'``) to count.
        """
        probs = np.abs(state) ** 2
        probs = np.real(probs)
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

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return measurement probabilities ``|ψᵢ|²`` for each
        computational basis state.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector of shape ``(2ⁿ,)``.

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(2ⁿ,)``.
        """
        return np.real(np.abs(state) ** 2)

    # ------------------------------------------------------------------
    # Circuit execution
    # ------------------------------------------------------------------

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Execute a quantum circuit and return the final statevector.

        Iterates through every operation in the circuit and applies it
        sequentially.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to simulate.
        initial_state : numpy.ndarray, optional
            Custom initial state.  Defaults to ``|00…0⟩``.

        Returns
        -------
        numpy.ndarray
            Final statevector of shape ``(2**n,)``.
        """
        n = circuit.num_qubits
        if initial_state is not None:
            state = initial_state.copy().astype(self._dtype)
        else:
            state = self.zero_state(n)

        for op in circuit.data:
            if isinstance(op, Barrier):
                continue
            if isinstance(op, Reset):
                state = self._apply_reset(state, op.qubits, n)
                continue
            if isinstance(op, (Operation,)):
                if isinstance(op.gate, Measurement):
                    state = self._apply_measurement_op(state, op.qubits, n)
                    continue
                self.apply_gate(state, op.gate, op.qubits, op.params)
                continue

        return state

    def _apply_reset(
        self,
        state: np.ndarray,
        qubits: Tuple[int, ...],
        n: int,
    ) -> np.ndarray:
        """Reset the specified qubit(s) to |0⟩.

        Implements a projective measurement discard: measure each qubit
        in the computational basis and force the result to 0, then
        renormalise.

        Parameters
        ----------
        state : numpy.ndarray
        qubits : tuple of int
        n : int

        Returns
        -------
        numpy.ndarray
        """
        dim = state.shape[0]
        for q in qubits:
            # Zero out all amplitudes where qubit q is |1⟩
            mask = np.array(
                [((idx >> (n - 1 - q)) & 1) == 0 for idx in range(dim)],
                dtype=bool,
            )
            state[~mask] = 0.0
            _normalize(state)
        return state

    def _apply_measurement_op(
        self,
        state: np.ndarray,
        qubits: Tuple[int, ...],
        n: int,
    ) -> np.ndarray:
        """Apply a measurement operation — collapses the state in-place.

        Parameters
        ----------
        state : numpy.ndarray
        qubits : tuple of int
        n : int

        Returns
        -------
        numpy.ndarray
        """
        _, state, _ = self.apply_measurement(state, list(qubits), n)
        return state

    # ------------------------------------------------------------------
    # Batch simulation
    # ------------------------------------------------------------------

    def run_circuit_batch(
        self,
        circuit: QuantumCircuit,
        batch_size: int,
        initial_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the same circuit on a batch of initial states.

        Parameters
        ----------
        circuit : QuantumCircuit
        batch_size : int
            Number of initial states in the batch.
        initial_states : numpy.ndarray, optional
            Shape ``(batch_size, 2**n)``.  If ``None``, all states are
            initialised to ``|00…0⟩``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch_size, 2**n)`` — the final statevectors.
        """
        n = circuit.num_qubits
        dim = 1 << n

        if initial_states is not None:
            batch = initial_states.copy().astype(self._dtype)
        else:
            batch = np.zeros((batch_size, dim), dtype=self._dtype)
            batch[:, 0] = 1.0

        # Pre-compute all gate matrices once
        gate_ops: List[Tuple[np.ndarray, Tuple[int, ...]]] = []
        for op in circuit.data:
            if isinstance(op, (Barrier, Reset)):
                continue
            if isinstance(op, Operation) and not isinstance(op.gate, Measurement):
                mat = self._get_gate_matrix(op.gate, op.params)
                gate_ops.append((mat, op.qubits))

        # Apply each gate across the batch using einsum
        for gate_matrix, qubits in gate_ops:
            batch = self._apply_gate_batch(batch, gate_matrix, qubits, n)

        # Normalise each row
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPS)
        batch /= norms

        return batch

    def _apply_gate_batch(
        self,
        batch: np.ndarray,
        gate_matrix: np.ndarray,
        qubits: Tuple[int, ...],
        n: int,
    ) -> np.ndarray:
        """Apply a gate to every state in the batch.

        Uses ``numpy.einsum`` over the reshaped
        ``(batch, 2, 2, …, 2)`` tensor.

        Parameters
        ----------
        batch : numpy.ndarray
            Shape ``(batch_size, 2**n)``.
        gate_matrix : numpy.ndarray
            Shape ``(2**k, 2**k)``.
        qubits : tuple of int
        n : int

        Returns
        -------
        numpy.ndarray
            Shape ``(batch_size, 2**n)``.
        """
        k = len(qubits)

        if k == n:
            return batch @ gate_matrix.T.conj()

        bs = batch.shape[0]
        shape = [2] * n
        tensor = batch.reshape([bs] + shape)

        gate_tensor = gate_matrix.reshape([2] * (2 * k))

        state_indices = list("abcdefghijklmnopqrstuvwxyz"[:n])
        batch_idx = "B"
        output_indices = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k])
        input_indices = list(state_indices[q] for q in qubits)

        new_state_indices = list(state_indices)
        for i, q in enumerate(qubits):
            new_state_indices[q] = output_indices[i]

        gate_subs = "".join(output_indices + input_indices)
        state_subs = batch_idx + "".join(state_indices)
        result_subs = batch_idx + "".join(new_state_indices)

        einsum_str = f"{gate_subs},{state_subs}->{result_subs}"

        sorted_positions = sorted(range(k), key=lambda i: qubits[i])
        perm = sorted_positions + [k + p for p in sorted_positions]
        gate_tensor_reordered = np.transpose(gate_tensor, perm)

        tensor = np.einsum(einsum_str, gate_tensor_reordered, tensor, optimize=True)
        return tensor.reshape(bs, -1)

    # ------------------------------------------------------------------
    # Gradients (for VQA)
    # ------------------------------------------------------------------

    def grad_params(
        self,
        circuit: QuantumCircuit,
        param_index: int,
        initial_state: Optional[np.ndarray] = None,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """Compute the gradient of the statevector with respect to a
        circuit parameter using the parameter-shift rule (for rotation
        gates) or central finite differences (fallback).

        The parameter is identified by its *index* among the ordered
        list of all parameterised operations in the circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
        param_index : int
            Index of the parameter to differentiate (0-based among all
            parameterised operations).
        initial_state : numpy.ndarray, optional
        eps : float, optional
            Step size for finite differences (fallback).

        Returns
        -------
        numpy.ndarray
            Gradient ``∂|ψ⟩/∂θᵢ``, shape ``(2ⁿ,)``.
        """
        # Collect all parameterised operations
        param_ops: List[Tuple[int, Operation]] = []
        for idx, op in enumerate(circuit.data):
            if isinstance(op, Operation) and op.params:
                param_ops.append((idx, op))

        if param_index < 0 or param_index >= len(param_ops):
            raise IndexError(
                f"param_index {param_index} out of range "
                f"(circuit has {len(param_ops)} parameterised operations)"
            )

        op_idx, op = param_ops[param_index]
        gate = op.gate

        # Try parameter-shift rule for standard rotation gates
        shift_result = self._try_parameter_shift(
            circuit, op_idx, op, initial_state
        )
        if shift_result is not None:
            return shift_result

        # Fallback: central finite difference
        return self._finite_difference_grad(
            circuit, op_idx, op, initial_state, eps
        )

    def _try_parameter_shift(
        self,
        circuit: QuantumCircuit,
        op_idx: int,
        op: Operation,
        initial_state: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Attempt the parameter-shift rule for rotation gates.

        Returns ``None`` if the gate does not support the shift rule.
        """
        gate_name = op.gate.name
        shift_gates = {"rx", "ry", "rz", "p", "u1", "crx", "cry", "crz", "rxx", "ryy", "rzz", "rzx", "cphase"}
        if gate_name not in shift_gates:
            return None

        theta = op.params[0]
        state_plus = self._run_with_shifted_param(
            circuit, op_idx, theta + math.pi / 2, initial_state
        )
        state_minus = self._run_with_shifted_param(
            circuit, op_idx, theta - math.pi / 2, initial_state
        )

        return 0.5 * (state_plus - state_minus)

    def _finite_difference_grad(
        self,
        circuit: QuantumCircuit,
        op_idx: int,
        op: Operation,
        initial_state: Optional[np.ndarray],
        eps: float,
    ) -> np.ndarray:
        """Central finite-difference gradient."""
        theta = op.params[0]
        state_plus = self._run_with_shifted_param(
            circuit, op_idx, theta + eps, initial_state
        )
        state_minus = self._run_with_shifted_param(
            circuit, op_idx, theta - eps, initial_state
        )
        return (state_plus - state_minus) / (2.0 * eps)

    def _run_with_shifted_param(
        self,
        circuit: QuantumCircuit,
        op_idx: int,
        new_value: float,
        initial_state: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run the circuit with one parameter replaced by *new_value*.

        Parameters
        ----------
        circuit : QuantumCircuit
        op_idx : int
            Index in ``circuit.data`` of the operation to modify.
        new_value : float
        initial_state : numpy.ndarray or None

        Returns
        -------
        numpy.ndarray
        """
        n = circuit.num_qubits
        state = (
            initial_state.copy().astype(self._dtype)
            if initial_state is not None
            else self.zero_state(n)
        )

        for i, op in enumerate(circuit.data):
            if isinstance(op, Barrier):
                continue
            if isinstance(op, Reset):
                state = self._apply_reset(state, op.qubits, n)
                continue
            if isinstance(op, Operation):
                if isinstance(op.gate, Measurement):
                    state = self._apply_measurement_op(state, op.qubits, n)
                    continue
                if i == op_idx and op.params:
                    # Use the shifted parameter value
                    self.apply_gate(state, op.gate, op.qubits, (new_value,))
                else:
                    self.apply_gate(state, op.gate, op.qubits, op.params)
                continue

        return state

    # ------------------------------------------------------------------
    # Expectation-value gradient
    # ------------------------------------------------------------------

    def expectation_grad(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        param_index: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> float:
        r"""Compute ``∂⟨ψ(θ)|O|ψ(θ)⟩/∂θᵢ``.

        Uses the parameter-shift rule for rotation gates, with a
        central-difference fallback.

        Parameters
        ----------
        circuit : QuantumCircuit
        observable : numpy.ndarray
            Shape ``(2ⁿ, 2ⁿ)``.
        param_index : int
        initial_state : numpy.ndarray, optional

        Returns
        -------
        float
        """
        grad_sv = self.grad_params(circuit, param_index, initial_state)
        # Compute final state for the current parameters
        final_state = self.run_circuit(circuit, initial_state)
        # d⟨O⟩ = 2 Re( ⟨dψ| O |ψ⟩ )  for a real observable
        O_psi = observable @ final_state
        result = 2.0 * np.real(np.vdot(grad_sv, O_psi))
        return float(result)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the gate-matrix cache."""
        self._gate_cache.clear()

    def statevector_from_array(
        self,
        arr: np.ndarray,
    ) -> Statevector:
        """Wrap a raw numpy array in a :class:`Statevector` object.

        Parameters
        ----------
        arr : numpy.ndarray
            Shape ``(2ⁿ,)``.

        Returns
        -------
        Statevector
        """
        return Statevector(arr, normalize=False)
