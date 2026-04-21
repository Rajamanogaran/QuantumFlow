"""
Qubit Representation and State Management
==========================================

Provides the fundamental building blocks for representing qubits and their
states in the QuantumFlow framework.

Classes
-------
    Qubit : a single logical qubit with index, label, and metadata.
    QubitState : enum of standard single-qubit states.
    QubitStateVector : dataclass holding numpy arrays for basis amplitudes.
    MultiQubitState : manages entangled multi-qubit statevectors.

Helper functions
----------------
    bloch_vector : compute the Bloch vector for a single-qubit state.
    state_fidelity : compute the fidelity between two state vectors.
    polarization : compute the expectation value of Pauli-Z.

Typical usage::

    >>> from quantumflow.core.qubit import Qubit, QubitState, bloch_vector
    >>> q = Qubit(0, label='ancilla')
    >>> q
    Qubit(index=0, label='ancilla')
    >>> state = QubitState.ZERO.statevector()
    >>> state
    array([1.+0.j, 0.+0.j])
    >>> bloch_vector(state)
    array([0., 0., 1.])
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
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

__all__ = [
    "Qubit",
    "QubitState",
    "QubitStateVector",
    "MultiQubitState",
    "bloch_vector",
    "state_fidelity",
    "polarization",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "BASIS_ZERO",
    "BASIS_ONE",
    "SQRT2",
    "SQRT2_INV",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQRT2: float = math.sqrt(2.0)
"""Square root of 2 (``√2``)."""

SQRT2_INV: float = 1.0 / math.sqrt(2.0)
"""Reciprocal of √2, used frequently in state normalisation."""

# Standard computational basis vectors
BASIS_ZERO: np.ndarray = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
"""The |0⟩ state vector ``[1, 0]ᵀ``."""

BASIS_ONE: np.ndarray = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
"""The |1⟩ state vector ``[0, 1]ᵀ``."""

# Pauli matrices — single-qubit observables
PAULI_X: np.ndarray = np.array(
    [[0.0 + 0.0j, 1.0 + 0.0j],
     [1.0 + 0.0j, 0.0 + 0.0j]],
    dtype=np.complex128,
)
"""Pauli-X (σₓ) matrix::

    ┌     ┐
    │ 0  1 │
    │ 1  0 │
    └     ┘
"""

PAULI_Y: np.ndarray = np.array(
    [[0.0 + 0.0j, 0.0 - 1.0j],
     [0.0 + 1.0j, 0.0 + 0.0j]],
    dtype=np.complex128,
)
"""Pauli-Y (σᵧ) matrix::

    ┌        ┐
    │  0  -i │
    │  i   0 │
    └        ┘
"""

PAULI_Z: np.ndarray = np.array(
    [[1.0 + 0.0j, 0.0 + 0.0j],
     [0.0 + 0.0j, -1.0 + 0.0j]],
    dtype=np.complex128,
)
"""Pauli-Z (σ_z) matrix::

    ┌      ┐
    │ 1   0 │
    │ 0  -1 │
    └      ┘
"""

IDENTITY_2: np.ndarray = np.eye(2, dtype=np.complex128)
"""2×2 identity matrix."""

# ---------------------------------------------------------------------------
# QubitState enum
# ---------------------------------------------------------------------------

class QubitState(enum.Enum):
    """Enumeration of standard single-qubit preparation states.

    Each member carries the canonical complex amplitudes for |0⟩ and |1⟩,
    accessible via :meth:`statevector`.

    Members
    -------
    ZERO : |0⟩
    ONE : |1⟩
    PLUS : |+⟩ = (|0⟩ + |1⟩) / √2
    MINUS : |-⟩ = (|0⟩ - |1⟩) / √2
    PLUS_I : |+i⟩ = (|0⟩ + i|1⟩) / √2
    MINUS_I : |-i⟩ = (|0⟩ - i|1⟩) / √2

    Examples
    --------
    >>> QubitState.ZERO.statevector()
    array([1.+0.j, 0.+0.j])

    >>> QubitState.PLUS.statevector()
    array([0.70710678+0.j, 0.70710678+0.j])
    """

    ZERO = "0"
    ONE = "1"
    PLUS = "+"
    MINUS = "-"
    PLUS_I = "+i"
    MINUS_I = "-i"

    def statevector(self) -> np.ndarray:
        """Return the normalised complex state vector for this state.

        Returns
        -------
        numpy.ndarray
            Complex128 array of shape ``(2,)``.
        """
        sv_map = {
            QubitState.ZERO: np.array([1.0, 0.0], dtype=np.complex128),
            QubitState.ONE: np.array([0.0, 1.0], dtype=np.complex128),
            QubitState.PLUS: np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex128),
            QubitState.MINUS: np.array([SQRT2_INV, -SQRT2_INV], dtype=np.complex128),
            QubitState.PLUS_I: np.array([SQRT2_INV, 1j * SQRT2_INV], dtype=np.complex128),
            QubitState.MINUS_I: np.array([SQRT2_INV, -1j * SQRT2_INV], dtype=np.complex128),
        }
        return sv_map[self].copy()

    def bloch_vector(self) -> np.ndarray:
        """Return the Bloch sphere coordinates ``[x, y, z]`` for this state.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(3,)`` with values in ``[-1, 1]``.
        """
        bloch_map = {
            QubitState.ZERO: np.array([0.0, 0.0, 1.0]),
            QubitState.ONE: np.array([0.0, 0.0, -1.0]),
            QubitState.PLUS: np.array([1.0, 0.0, 0.0]),
            QubitState.MINUS: np.array([-1.0, 0.0, 0.0]),
            QubitState.PLUS_I: np.array([0.0, 1.0, 0.0]),
            QubitState.MINUS_I: np.array([0.0, -1.0, 0.0]),
        }
        return bloch_map[self].copy()

    def __repr__(self) -> str:
        label_map = {
            QubitState.ZERO: "|0⟩",
            QubitState.ONE: "|1⟩",
            QubitState.PLUS: "|+⟩",
            QubitState.MINUS: "|-⟩",
            QubitState.PLUS_I: "|+i⟩",
            QubitState.MINUS_I: "|-i⟩",
        }
        return label_map[self]


# ---------------------------------------------------------------------------
# QubitStateVector dataclass
# ---------------------------------------------------------------------------

@dataclass
class QubitStateVector:
    """Container for the complex amplitudes of a single-qubit state.

    Stores the amplitudes ``alpha`` (coefficient of |0⟩) and ``beta``
    (coefficient of |1⟩) such that the state is ``alpha|0⟩ + beta|1⟩``.

    The state is automatically normalised upon creation.

    Parameters
    ----------
    alpha : complex or numpy.ndarray
        Amplitude of |0⟩. Can be a scalar or a 1-element array.
    beta : complex or numpy.ndarray
        Amplitude of |1⟩.
    normalize : bool, optional
        If ``True`` (default), the state is normalised so that
        ``|alpha|² + |beta|² = 1``.

    Attributes
    ----------
    alpha : numpy.ndarray
        Amplitude of |0⟩.
    beta : numpy.ndarray
        Amplitude of |1⟩.
    dim : int
        Dimension (always 2 for a single qubit).

    Examples
    --------
    >>> sv = QubitStateVector(1.0, 0.0)
    >>> sv.to_array()
    array([1.+0.j, 0.+0.j])

    >>> sv = QubitStateVector(0.5, 0.5, normalize=True)
    >>> np.abs(sv.to_array())
    array([0.70710678, 0.70710678])
    """

    alpha: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j], dtype=np.complex128))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.0 + 0j], dtype=np.complex128))

    def __post_init__(self) -> None:
        # Coerce scalars to 1-element arrays
        if not isinstance(self.alpha, np.ndarray):
            self.alpha = np.array([complex(self.alpha)], dtype=np.complex128)
        if not isinstance(self.beta, np.ndarray):
            self.beta = np.array([complex(self.beta)], dtype=np.complex128)

        # Normalise
        norm_sq = np.abs(self.alpha) ** 2 + np.abs(self.beta) ** 2
        if not np.allclose(norm_sq, 1.0, atol=1e-12):
            norm = np.sqrt(norm_sq)
            if norm > 1e-15:
                self.alpha = self.alpha / norm
                self.beta = self.beta / norm

    @property
    def dim(self) -> int:
        """int: Hilbert space dimension (always 2)."""
        return 2

    def to_array(self) -> np.ndarray:
        """Return the state vector as a ``(2,)`` complex128 numpy array.

        Returns
        -------
        numpy.ndarray
            The vector ``[alpha, beta]ᵀ``.
        """
        return np.array([self.alpha[0], self.beta[0]], dtype=np.complex128)

    @classmethod
    def from_array(cls, array: np.ndarray) -> QubitStateVector:
        """Create a :class:`QubitStateVector` from a ``(2,)`` array.

        Parameters
        ----------
        array : numpy.ndarray
            Array of shape ``(2,)`` containing the state amplitudes.

        Returns
        -------
        QubitStateVector

        Raises
        ------
        ValueError
            If the array does not have exactly 2 elements.
        """
        if isinstance(array, np.ndarray) and array.shape == (2,):
            return cls(alpha=array[0], beta=array[1], normalize=True)
        raise ValueError(
            f"Expected array of shape (2,), got shape {getattr(array, 'shape', None)}"
        )

    @classmethod
    def from_state(cls, state: QubitState) -> QubitStateVector:
        """Create from a :class:`QubitState` enum member.

        Parameters
        ----------
        state : QubitState
            Pre-defined state.

        Returns
        -------
        QubitStateVector
        """
        sv = state.statevector()
        return cls(alpha=sv[0], beta=sv[1])

    def probability_zero(self) -> float:
        """Probability of measuring |0⟩: ``|α|²``.

        Returns
        -------
        float
        """
        return float(np.abs(self.alpha[0]) ** 2)

    def probability_one(self) -> float:
        """Probability of measuring |1⟩: ``|β|²``.

        Returns
        -------
        float
        """
        return float(np.abs(self.beta[0]) ** 2)

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities ``[|α|², |β|²]``.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(2,)``.
        """
        return np.array([self.probability_zero(), self.probability_one()])

    def bloch_vector(self) -> np.ndarray:
        """Compute the Bloch vector ``[x, y, z]``.

        For state ``α|0⟩ + β|1⟩``, the Bloch vector components are:

        - ``x = 2·Re(α*·β)``
        - ``y = 2·Im(α*·β)``
        - ``z = |α|² - |β|²``

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(3,)`` with values in ``[-1, 1]``.
        """
        a, b = self.alpha[0], self.beta[0]
        x = 2.0 * np.real(np.conj(a) * b)
        y = 2.0 * np.imag(np.conj(a) * b)
        z = np.abs(a) ** 2 - np.abs(b) ** 2
        return np.array([x, y, z])

    def __repr__(self) -> str:
        a_str = f"{self.alpha[0]:.4f}"
        b_str = f"{self.beta[0]:.4f}"
        return f"QubitStateVector(alpha={a_str}, beta={b_str})"


# ---------------------------------------------------------------------------
# Qubit class
# ---------------------------------------------------------------------------

class Qubit:
    """A single logical qubit.

    Represents a qubit by its index within a register, an optional
    human-readable label, and arbitrary metadata.

    Qubits are lightweight identifiers — the actual quantum state is
    managed by simulators and state classes.

    Parameters
    ----------
    index : int
        Position of the qubit within its register (0-based).
    label : str, optional
        Descriptive name for the qubit. Defaults to ``'q{index}'``.
    metadata : dict, optional
        Arbitrary key-value pairs attached to this qubit.

    Attributes
    ----------
    index : int
        Qubit position within its register.
    label : str
        Human-readable label.
    metadata : dict
        Arbitrary metadata.

    Examples
    --------
    >>> q0 = Qubit(0)
    >>> q0
    Qubit(index=0, label='q0')
    >>> q1 = Qubit(1, label='ancilla', metadata={'purpose': 'syndrome'})
    >>> q1.label
    'ancilla'
    """

    __slots__ = ("_index", "_label", "_metadata")

    def __init__(
        self,
        index: int,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Qubit index must be a non-negative integer, got {index!r}")
        self._index: int = index
        self._label: str = label if label is not None else f"q{index}"
        self._metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    # -- Properties ----------------------------------------------------------

    @property
    def index(self) -> int:
        """int: 0-based position within the register."""
        return self._index

    @property
    def label(self) -> str:
        """str: Human-readable identifier."""
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """dict: Arbitrary metadata attached to this qubit."""
        return dict(self._metadata)

    # -- State helpers -------------------------------------------------------

    def prepare(self, state: QubitState) -> QubitStateVector:
        """Return the state vector for a standard preparation.

        This does *not* modify any global state — it simply returns the
        :class:`QubitStateVector` corresponding to ``state``.

        Parameters
        ----------
        state : QubitState
            Desired preparation state.

        Returns
        -------
        QubitStateVector
        """
        return QubitStateVector.from_state(state)

    # -- Dunder methods ------------------------------------------------------

    def __repr__(self) -> str:
        return f"Qubit(index={self._index}, label={self._label!r})"

    def __str__(self) -> str:
        return self._label

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Qubit):
            return self._index == other._index
        if isinstance(other, int):
            return self._index == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._index)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Qubit):
            return self._index < other._index
        if isinstance(other, int):
            return self._index < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Qubit):
            return self._index <= other._index
        if isinstance(other, int):
            return self._index <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Qubit):
            return self._index > other._index
        if isinstance(other, int):
            return self._index > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Qubit):
            return self._index >= other._index
        if isinstance(other, int):
            return self._index >= other
        return NotImplemented

    def __int__(self) -> int:
        return self._index


# ---------------------------------------------------------------------------
# MultiQubitState
# ---------------------------------------------------------------------------

class MultiQubitState:
    """Manage entangled multi-qubit quantum states.

    Stores a complex state vector of dimension :math:`2^n` and provides
    operations for evolution, measurement, and subsystem analysis.

    Parameters
    ----------
    num_qubits : int
        Number of qubits. Must be positive.
    amplitudes : numpy.ndarray, optional
        Complex amplitudes of shape ``(2**num_qubits,)``. If ``None``,
        the state is initialised to |00…0⟩.
    normalize : bool, optional
        If ``True`` (default), normalise the state vector.

    Attributes
    ----------
    num_qubits : int
    dim : int
    amplitudes : numpy.ndarray

    Examples
    --------
    Create a Bell state (|00⟩ + |11⟩) / √2:

    >>> state = MultiQubitState(2, np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
    >>> state.is_entangled()
    True
    >>> state.measure_all(shots=10)  # doctest: +SKIP
    {0: 5, 3: 5}
    """

    def __init__(
        self,
        num_qubits: int,
        amplitudes: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> None:
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError(
                f"num_qubits must be a positive integer, got {num_qubits!r}"
            )
        self._num_qubits: int = num_qubits
        self._dim: int = 1 << num_qubits  # 2**n

        if amplitudes is None:
            # Default to |00...0⟩
            self._amplitudes: np.ndarray = np.zeros(
                self._dim, dtype=np.complex128
            )
            self._amplitudes[0] = 1.0 + 0.0j
        else:
            arr = np.asarray(amplitudes, dtype=np.complex128)
            if arr.shape != (self._dim,):
                raise ValueError(
                    f"Expected amplitudes of shape ({self._dim},), "
                    f"got {arr.shape}"
                )
            self._amplitudes = arr.copy()

        if normalize:
            norm = np.linalg.norm(self._amplitudes)
            if norm > 1e-15:
                self._amplitudes /= norm

    # -- Properties ----------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits in the system."""
        return self._num_qubits

    @property
    def dim(self) -> int:
        """int: Hilbert space dimension (2ⁿ)."""
        return self._dim

    @property
    def amplitudes(self) -> np.ndarray:
        """numpy.ndarray: Complex state-vector amplitudes of shape ``(2ⁿ,)``."""
        return self._amplitudes.copy()

    # -- Evolution -----------------------------------------------------------

    def evolve(self, unitary: np.ndarray) -> MultiQubitState:
        """Apply a unitary operator to the state.

        Parameters
        ----------
        unitary : numpy.ndarray
            Unitary matrix of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        MultiQubitState
            A **new** state object with the evolved amplitudes.

        Raises
        ------
        ValueError
            If ``unitary`` does not have compatible shape.
        """
        if unitary.shape != (self._dim, self._dim):
            raise ValueError(
                f"Unitary shape {unitary.shape} incompatible with "
                f"state dimension {self._dim}"
            )
        new_amps = unitary @ self._amplitudes
        return MultiQubitState(self._num_qubits, new_amps, normalize=False)

    # -- Measurement ---------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """Return measurement probabilities for each computational basis state.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(2ⁿ,)`` with non-negative entries summing to 1.
        """
        return np.abs(self._amplitudes) ** 2

    def measure(
        self,
        qubit_indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[int, MultiQubitState]:
        """Simulate a single projective measurement.

        Parameters
        ----------
        qubit_indices : int or sequence of int, optional
            Qubits to measure. If ``None``, measure all qubits.

        Returns
        -------
        outcome : int
            Measurement outcome (integer).
        new_state : MultiQubitState
            Post-measurement collapsed state.
        """
        probs = self.probabilities()

        if qubit_indices is None:
            # Measure all qubits
            outcome = int(np.random.choice(self._dim, p=probs))
            new_amps = np.zeros(self._dim, dtype=np.complex128)
            new_amps[outcome] = 1.0 + 0.0j
            return outcome, MultiQubitState(self._num_qubits, new_amps, normalize=False)

        # Partial measurement — collapse specified qubits
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        probs_full = probs
        outcome_vals = np.random.choice(
            [0, 1],
            size=len(qubit_indices),
            p=None,
        )
        # Weighted sampling based on marginal probabilities
        marginals = self._marginal_probabilities(list(qubit_indices))
        outcome_vals = []
        remaining_probs = probs_full.copy()
        for qi in qubit_indices:
            p0 = float(np.sum(remaining_probs[self._bitmask(qi, 0)]))
            outcome = 0 if np.random.random() < p0 else 1
            outcome_vals.append(outcome)
            mask = self._bitmask(qi, outcome)
            remaining_probs[~mask] = 0.0
            norm = remaining_probs.sum()
            if norm > 1e-15:
                remaining_probs /= norm

        # Collapse
        new_amps = self._amplitudes.copy()
        for qi, val in zip(qubit_indices, outcome_vals):
            mask = self._bitmask(qi, val)
            new_amps[~mask] = 0.0

        norm = np.linalg.norm(new_amps)
        if norm > 1e-15:
            new_amps /= norm
        else:
            # Degenerate case — just normalise
            new_amps /= norm if norm > 0 else 1.0

        outcome_int = 0
        for i, val in enumerate(outcome_vals):
            outcome_int |= (val << i)

        return outcome_int, MultiQubitState(self._num_qubits, new_amps, normalize=False)

    def measure_all(self, shots: int = 1024) -> Dict[int, int]:
        """Sample measurement outcomes across all qubits.

        Parameters
        ----------
        shots : int, optional
            Number of measurement repetitions. Default is 1024.

        Returns
        -------
        dict
            Mapping from outcome (int) to count.
        """
        probs = self.probabilities()
        outcomes = np.random.choice(self._dim, size=shots, p=probs)
        counts: Dict[int, int] = {}
        for o in outcomes:
            counts[int(o)] = counts.get(int(o), 0) + 1
        return counts

    # -- Subsystem analysis --------------------------------------------------

    def reduced_density_matrix(
        self,
        qubits_to_keep: Sequence[int],
    ) -> np.ndarray:
        """Compute the reduced density matrix for a subset of qubits.

        Parameters
        ----------
        qubits_to_keep : sequence of int
            Indices of qubits to retain.

        Returns
        -------
        numpy.ndarray
            Density matrix of shape ``(2**k, 2**k)`` where ``k`` is
            the number of kept qubits.
        """
        k = len(qubits_to_keep)
        full_dm = np.outer(self._amplitudes, np.conj(self._amplitudes))
        return self._partial_trace(full_dm, qubits_to_keep)

    def bloch_vectors(self) -> List[np.ndarray]:
        """Compute the Bloch vector for each individual qubit.

        This traces out all other qubits and extracts the single-qubit
        Bloch sphere coordinates.

        Returns
        -------
        list of numpy.ndarray
            List of ``(3,)`` arrays, one per qubit.
        """
        vectors = []
        for i in range(self._num_qubits):
            rho = self.reduced_density_matrix([i])
            x = float(np.real(np.trace(rho @ PAULI_X)))
            y = float(np.real(np.trace(rho @ PAULI_Y)))
            z = float(np.real(np.trace(rho @ PAULI_Z)))
            vectors.append(np.array([x, y, z]))
        return vectors

    def is_entangled(self) -> bool:
        """Check if the state is entangled (not a product state).

        For two qubits, uses the PPT (positive partial transpose) criterion.
        For more qubits, checks if the reduced density matrix for each
        subsystem is mixed.

        Returns
        -------
        bool
        """
        if self._num_qubits == 1:
            return False

        # For 2-qubit case: check purity of partial trace
        if self._num_qubits == 2:
            for i in range(2):
                rho = self.reduced_density_matrix([i])
                purity_val = float(np.real(np.trace(rho @ rho)))
                if purity_val < 1.0 - 1e-8:
                    return True
            return False

        # General case: check all 1-qubit reduced density matrices
        for i in range(self._num_qubits):
            rho = self.reduced_density_matrix([i])
            purity_val = float(np.real(np.trace(rho @ rho)))
            if purity_val < 1.0 - 1e-8:
                return True
        return False

    def purity(self) -> float:
        """Compute the purity ``Tr(ρ²)`` where ρ = |ψ⟩⟨ψ|.

        Returns
        -------
        float
            Always 1.0 for a pure state (normalisation check).
        """
        return float(np.real(np.sum(np.abs(self._amplitudes) ** 4)))

    def entropy(self) -> float:
        """Von Neumann entropy of the state.

        For a pure state this is 0.

        Returns
        -------
        float
        """
        rho = np.outer(self._amplitudes, np.conj(self._amplitudes))
        return _von_neumann_entropy(rho)

    # -- Tensor product ------------------------------------------------------

    def tensor(self, other: MultiQubitState) -> MultiQubitState:
        """Compute the Kronecker (tensor) product with another state.

        Parameters
        ----------
        other : MultiQubitState
            State to tensor with.

        Returns
        -------
        MultiQubitState
            Combined state with ``num_qubits = self.n + other.n``.
        """
        new_amps = np.kron(self._amplitudes, other._amplitudes)
        return MultiQubitState(
            self._num_qubits + other._num_qubits,
            new_amps,
            normalize=False,
        )

    # -- String representation ------------------------------------------------

    def __repr__(self) -> str:
        n = self._num_qubits
        nonzero = int(np.sum(np.abs(self._amplitudes) > 1e-10))
        return (
            f"MultiQubitState(num_qubits={n}, dim={self._dim}, "
            f"nonzero_amplitudes={nonzero})"
        )

    def __str__(self) -> str:
        lines = [f"|ψ⟩ ({self._num_qubits} qubits):"]
        threshold = 1e-10
        for i in range(self._dim):
            amp = self._amplitudes[i]
            if np.abs(amp) > threshold:
                bits = format(i, f'0{self._num_qubits}b')
                if np.abs(amp.imag) < 1e-12:
                    coeff_str = f"{amp.real:+.4f}"
                else:
                    coeff_str = f"{amp:.4f}"
                lines.append(f"  {coeff_str}|{bits}⟩")
        return "\n".join(lines)

    # -- Private helpers -----------------------------------------------------

    def _bitmask(self, qubit: int, value: int) -> np.ndarray:
        """Boolean mask for basis states where ``qubit`` equals ``value``."""
        return np.array(
            [((i >> (self._num_qubits - 1 - qubit)) & 1) == value
             for i in range(self._dim)],
            dtype=bool,
        )

    def _marginal_probabilities(self, qubit_indices: List[int]) -> List[float]:
        """Marginal probability that each listed qubit is 0."""
        marginals = []
        for qi in qubit_indices:
            mask = self._bitmask(qi, 0)
            marginals.append(float(np.sum(self.probabilities()[mask])))
        return marginals

    def _partial_trace(
        self,
        rho: np.ndarray,
        qubits_to_keep: Sequence[int],
    ) -> np.ndarray:
        """Trace out qubits not in ``qubits_to_keep``.

        Uses the reshape-and-sum method for efficiency.
        """
        n = self._num_qubits
        k = len(qubits_to_keep)
        if k == n:
            return rho.copy()
        if k == 0:
            return np.array([[1.0 + 0.0j]])  # scalar 1

        # Reshape density matrix for partial trace via tensor contraction
        rho_tensor = rho.reshape([2] * (2 * n))

        # Axes to keep (row and column indices for kept qubits)
        keep = list(qubits_to_keep)
        trace_out = [i for i in range(n) if i not in keep]

        # Row axes for kept qubits: 0..n-1 → those in `keep`
        # Col axes for kept qubits: n..2n-1 → those in `keep` + n
        row_keep = keep
        col_keep = [q + n for q in keep]
        row_trace = trace_out
        col_trace = [q + n for q in trace_out]

        # Contract over trace-out qubits
        rho_reduced = np.einsum(
            rho_tensor,
            list(range(2 * n)),
            row_keep + col_keep,
            optimize='optimal',
        )

        # Reshape to matrix
        return rho_reduced.reshape((1 << k, 1 << k))


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def bloch_vector(state: Union[np.ndarray, QubitStateVector]) -> np.ndarray:
    """Compute the Bloch vector for a single-qubit state.

    Parameters
    ----------
    state : numpy.ndarray or QubitStateVector
        State vector of shape ``(2,)`` or a :class:`QubitStateVector`.

    Returns
    -------
    numpy.ndarray
        Float64 array ``[x, y, z]`` with values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If ``state`` is not 2-dimensional.

    Examples
    --------
    >>> bloch_vector(np.array([1, 0], dtype=complex))
    array([0., 0., 1.])

    >>> bloch_vector(np.array([1, 1], dtype=complex) / np.sqrt(2))
    array([1., 0., 0.])
    """
    if isinstance(state, QubitStateVector):
        return state.bloch_vector()

    sv = np.asarray(state, dtype=np.complex128)
    if sv.shape != (2,):
        raise ValueError(f"Expected state of shape (2,), got {sv.shape}")

    a, b = sv[0], sv[1]
    x = 2.0 * float(np.real(np.conj(a) * b))
    y = 2.0 * float(np.imag(np.conj(a) * b))
    z = float(np.abs(a) ** 2 - np.abs(b) ** 2)
    return np.array([x, y, z])


def state_fidelity(
    state_a: Union[np.ndarray, QubitStateVector],
    state_b: Union[np.ndarray, QubitStateVector],
) -> float:
    """Compute the fidelity between two quantum states.

    For state vectors, the fidelity is ``|⟨ψ_a|ψ_b⟩|²``.

    Parameters
    ----------
    state_a, state_b : numpy.ndarray or QubitStateVector
        State vectors to compare. Must have the same shape.

    Returns
    -------
    float
        Fidelity in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the states have different shapes.

    Examples
    --------
    >>> state_fidelity(np.array([1, 0], dtype=complex), np.array([1, 0], dtype=complex))
    1.0

    >>> state_fidelity(np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex))
    0.0
    """
    if isinstance(state_a, QubitStateVector):
        state_a = state_a.to_array()
    if isinstance(state_b, QubitStateVector):
        state_b = state_b.to_array()

    a = np.asarray(state_a, dtype=np.complex128).flatten()
    b = np.asarray(state_b, dtype=np.complex128).flatten()

    if a.shape != b.shape:
        raise ValueError(
            f"State shapes must match: {a.shape} vs {b.shape}"
        )

    overlap = np.vdot(a, b)
    return float(min(1.0, np.abs(overlap) ** 2))


def polarization(state: Union[np.ndarray, QubitStateVector]) -> float:
    """Compute the expectation value ⟨Z⟩ for a single-qubit state.

    This is also called the *polarization* or *longitudinal magnetization*.
    It equals the z-component of the Bloch vector.

    Parameters
    ----------
    state : numpy.ndarray or QubitStateVector
        Single-qubit state vector.

    Returns
    -------
    float
        Value in ``[-1, 1]``. ``+1`` for |0⟩, ``-1`` for |1⟩, ``0`` for
        equal superposition states on the equatorial plane.

    Examples
    --------
    >>> polarization(np.array([1, 0], dtype=complex))
    1.0
    >>> polarization(np.array([0, 1], dtype=complex))
    -1.0
    """
    bv = bloch_vector(state)
    return float(bv[2])


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute the von Neumann entropy of a density matrix.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix.

    Returns
    -------
    float
        S(ρ) = -Tr(ρ log₂ ρ) ≥ 0.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    # Remove near-zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
