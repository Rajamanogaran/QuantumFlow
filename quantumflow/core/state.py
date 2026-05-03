"""
Quantum State Representations
==============================

Provides classes for representing and manipulating quantum states:

* :class:`Statevector` — pure-state vector ``|ψ⟩`` of shape ``(2ⁿ,)``.
* :class:`DensityMatrix` — general mixed state ``ρ`` of shape ``(2ⁿ, 2ⁿ)``.
* :class:`Operator` / :class:`Observable` — Hermitian operators on the
  Hilbert space.

All classes use ``numpy`` complex128 arrays and provide a comprehensive
set of linear-algebra operations (evolution, measurement, partial trace,
fidelity, entropy, …).

Typical usage::

    >>> from quantumflow.core.state import Statevector, DensityMatrix
    >>> sv = Statevector.zero(2)          # |00⟩
    >>> probs = sv.probabilities()         # [1, 0, 0, 0]
    >>> dm = sv.to_density_matrix()
    >>> dm.is_pure()
    True

    >>> sv2 = Statevector.from_label("01+")  # |01+⟩ = |01⟩ ⊗ (|0⟩+|1⟩)/√2
    >>> sv2.dim
    8
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

import numpy as np

__all__ = [
    "QuantumState",
    "Statevector",
    "DensityMatrix",
    "Operator",
    "Observable",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEX_DTYPE = np.complex128
_FLOAT_DTYPE = np.float64
_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class QuantumState(ABC):
    """Abstract base class for quantum state representations.

    Subclasses must implement :meth:`dim`, :meth:`num_qubits`,
    :meth:`copy`, and :meth:`to_density_matrix`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """int: Hilbert space dimension (2ⁿ)."""
        ...

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """int: Number of qubits."""
        ...

    @abstractmethod
    def copy(self) -> QuantumState:
        """Return a deep copy."""
        ...

    @abstractmethod
    def to_density_matrix(self) -> DensityMatrix:
        """Convert to a density matrix representation."""
        ...

    # -- Common operations ---------------------------------------------------

    def is_pure(self) -> bool:
        """Whether this state is a pure state.

        The default implementation converts to a density matrix and
        checks its purity.
        """
        dm = self.to_density_matrix()
        return dm.is_pure()


# ---------------------------------------------------------------------------
# Statevector
# ---------------------------------------------------------------------------

class Statevector(QuantumState):
    """Pure quantum state represented as a complex state vector.

    The state vector ``|ψ⟩`` is stored as a 1-D numpy array of shape
    ``(2ⁿ,)`` where ``n`` is the number of qubits.

    Parameters
    ----------
    data : array_like
        Complex amplitudes. If 1-D of length ``2ⁿ``, interpreted directly.
        If 2-D of shape ``(1, 2ⁿ)``, squeezed.
    dims : int or tuple of int, optional
        Explicit subsystem dimensions. If an int, all subsystems have
        the same dimension. If ``None``, inferred from data length.
    normalize : bool, optional
        Normalise on creation. Default ``True``.
    copy : bool, optional
        Copy the input array. Default ``True``.

    Attributes
    ----------
    data : numpy.ndarray
    dim : int
    num_qubits : int

    Examples
    --------
    >>> sv = Statevector([1, 0, 0, 0])  # |00⟩
    >>> sv.num_qubits
    2
    >>> sv.probabilities()
    array([1., 0., 0., 0.])

    >>> bell = Statevector([1, 0, 0, 1]) / np.sqrt(2)
    >>> bell.measure()
    (0, Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]))  # example outcome

    >>> sv_plus = Statevector.from_label("+")
    >>> sv_plus.dim
    2
    """

    def __init__(
        self,
        data: Union[np.ndarray, Sequence, Sequence[Sequence]],
        dims: Optional[Union[int, Tuple[int, ...]]] = None,
        normalize: bool = True,
        copy: bool = True,
    ) -> None:
        arr = np.asarray(data, dtype=_COMPLEX_DTYPE)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1-D array, got shape {arr.shape}"
            )

        if copy:
            arr = arr.copy()

        self._data: np.ndarray = arr
        self._dim: int = arr.shape[0]
        self._num_qubits: int = int(round(math.log2(self._dim))) if self._dim > 0 else 0

        # Validate power-of-2 dimension
        if (1 << self._num_qubits) != self._dim:
            raise ValueError(
                f"Dimension must be a power of 2, got {self._dim}"
            )

        if dims is not None:
            if isinstance(dims, int):
                if dims != 2:
                    raise NotImplementedError("Only qubit (d=2) subsystems supported")
            else:
                if tuple(dims) != (2,) * self._num_qubits:
                    raise NotImplementedError("Only qubit subsystems supported")

        if normalize:
            norm = np.linalg.norm(self._data)
            if norm > _TOLERANCE:
                self._data /= norm
            elif not np.allclose(self._data, 0, atol=_TOLERANCE):
                raise ValueError("Cannot normalise a non-zero state vector")

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def zero(cls, num_qubits: int) -> Statevector:
        """Create the |00…0⟩ state.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        Statevector
        """
        dim = 1 << num_qubits
        data = np.zeros(dim, dtype=_COMPLEX_DTYPE)
        data[0] = 1.0
        return cls(data, normalize=False, copy=False)

    @classmethod
    def one(cls, num_qubits: int) -> Statevector:
        """Create the |11…1⟩ state.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        Statevector
        """
        dim = 1 << num_qubits
        data = np.zeros(dim, dtype=_COMPLEX_DTYPE)
        data[-1] = 1.0
        return cls(data, normalize=False, copy=False)

    @classmethod
    def from_label(cls, label: str) -> Statevector:
        """Create a state vector from a ket label string.

        Supported characters:
        * ``0``, ``1`` — computational basis states
        * ``+`` — |+⟩ = (|0⟩+|1⟩)/√2
        * ``-`` — |-⟩ = (|0⟩-|1⟩)/√2
        * ``r`` — |+i⟩ = (|0⟩+i|1⟩)/√2
        * ``l`` — |-i⟩ = (|0⟩-i|1⟩)/√2

        Parameters
        ----------
        label : str
            Ket label, e.g. ``"01+"`` for |01+⟩.

        Returns
        -------
        Statevector

        Raises
        ------
        ValueError
            If the label contains unsupported characters.

        Examples
        --------
        >>> Statevector.from_label("01")
        Statevector(num_qubits=2, dim=4)

        >>> Statevector.from_label("0+")
        Statevector(num_qubits=2, dim=4)
        """
        label = label.strip()
        if not label:
            raise ValueError("Label must be a non-empty string")

        # Map characters to state vectors
        _SQRT2_INV = 1.0 / math.sqrt(2.0)
        char_map = {
            "0": np.array([1.0, 0.0], dtype=_COMPLEX_DTYPE),
            "1": np.array([0.0, 1.0], dtype=_COMPLEX_DTYPE),
            "+": np.array([_SQRT2_INV, _SQRT2_INV], dtype=_COMPLEX_DTYPE),
            "-": np.array([_SQRT2_INV, -_SQRT2_INV], dtype=_COMPLEX_DTYPE),
            "r": np.array([_SQRT2_INV, 1j * _SQRT2_INV], dtype=_COMPLEX_DTYPE),
            "l": np.array([_SQRT2_INV, -1j * _SQRT2_INV], dtype=_COMPLEX_DTYPE),
        }

        for ch in label:
            if ch not in char_map:
                raise ValueError(
                    f"Unsupported character '{ch}' in label '{label}'. "
                    f"Supported: 0, 1, +, -, r, l"
                )

        result = char_map[label[0]].copy()
        for ch in label[1:]:
            result = np.kron(result, char_map[ch])

        return cls(result, normalize=False, copy=False)

    @classmethod
    def random(cls, num_qubits: int, seed: Optional[int] = None) -> Statevector:
        """Create a random normalised state vector.

        Parameters
        ----------
        num_qubits : int
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Statevector
        """
        rng = np.random.default_rng(seed)
        dim = 1 << num_qubits
        # Generate random complex vector with Gaussian entries
        real = rng.standard_normal(dim)
        imag = rng.standard_normal(dim)
        data = real + 1j * imag
        data /= np.linalg.norm(data)
        return cls(data, normalize=False, copy=False)

    # -- Properties ----------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """numpy.ndarray: Copy of the state-vector amplitudes."""
        return self._data.copy()

    @property
    def dim(self) -> int:
        """int: Hilbert space dimension 2ⁿ."""
        return self._dim

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits."""
        return self._num_qubits

    # -- Probabilities & measurement -----------------------------------------

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities for each computational basis state.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(2ⁿ,)`` with non-negative entries
            summing to 1.
        """
        return np.abs(self._data) ** 2

    def probabilities_dict(self) -> Dict[str, float]:
        """Probabilities as a dictionary keyed by binary strings.

        Returns
        -------
        dict
            Mapping ``{'00': 0.5, '11': 0.5, ...}`` with entries only
            for non-zero probabilities.
        """
        probs = self.probabilities()
        result: Dict[str, float] = {}
        for i, p in enumerate(probs):
            if p > _TOLERANCE:
                bits = format(i, f"0{self._num_qubits}b")
                result[bits] = float(p)
        return result

    def measure(
        self,
        qubits: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[int, Statevector]:
        """Simulate a single projective measurement.

        Parameters
        ----------
        qubits : int or sequence of int, optional
            Qubits to measure. If ``None``, measure all qubits.

        Returns
        -------
        outcome : int
            Measurement outcome as an integer.
        new_state : Statevector
            Post-measurement collapsed state.
        """
        if qubits is None:
            return self._measure_all()

        if isinstance(qubits, int):
            qubits = [qubits]
        qubits = list(qubits)

        probs = self.probabilities()
        # Sample outcomes for specified qubits
        outcome_bits = []
        remaining = probs.copy()

        for qi in qubits:
            # Compute marginal probability of qubit qi being 0
            mask_0 = np.array(
                [((idx >> (self._num_qubits - 1 - qi)) & 1) == 0
                 for idx in range(self._dim)],
                dtype=bool,
            )
            p0 = float(np.sum(remaining[mask_0]))
            outcome = 0 if np.random.random() < p0 else 1
            outcome_bits.append(outcome)

            # Zero out incompatible amplitudes
            mask_outcome = np.array(
                [((idx >> (self._num_qubits - 1 - qi)) & 1) == outcome
                 for idx in range(self._dim)],
                dtype=bool,
            )
            remaining[~mask_outcome] = 0.0
            norm = remaining.sum()
            if norm > _TOLERANCE:
                remaining /= norm

        # Collapse state vector
        new_data = self._data.copy()
        for qi, val in zip(qubits, outcome_bits):
            mask = np.array(
                [((idx >> (self._num_qubits - 1 - qi)) & 1) == val
                 for idx in range(self._dim)],
                dtype=bool,
            )
            new_data[~mask] = 0.0

        norm = np.linalg.norm(new_data)
        if norm > _TOLERANCE:
            new_data /= norm

        outcome_int = 0
        for i, val in enumerate(outcome_bits):
            outcome_int |= (val << i)

        return outcome_int, Statevector(new_data, normalize=False, copy=False)

    def _measure_all(self) -> Tuple[int, Statevector]:
        """Measure all qubits at once."""
        probs = self.probabilities()
        outcome = int(np.random.choice(self._dim, p=probs))
        new_data = np.zeros(self._dim, dtype=_COMPLEX_DTYPE)
        new_data[outcome] = 1.0
        return outcome, Statevector(new_data, normalize=False, copy=False)

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Sample multiple measurement outcomes.

        Parameters
        ----------
        shots : int, optional
            Number of repetitions. Default 1024.

        Returns
        -------
        dict
            Mapping from binary string to count.
        """
        probs = self.probabilities()
        outcomes = np.random.choice(self._dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            bits = format(int(o), f"0{self._num_qubits}b")
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    # -- Evolution ------------------------------------------------------------

    def evolve(self, unitary: np.ndarray) -> Statevector:
        """Apply a unitary operator: ``|ψ'⟩ = U|ψ⟩``.

        Parameters
        ----------
        unitary : numpy.ndarray
            Unitary matrix of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        Statevector
            New state vector.
        """
        U = np.asarray(unitary, dtype=_COMPLEX_DTYPE)
        if U.shape != (self._dim, self._dim):
            raise ValueError(
                f"Unitary shape {U.shape} incompatible with state dim {self._dim}"
            )
        new_data = U @ self._data
        return Statevector(new_data, normalize=False, copy=False)

    # -- Expectation values --------------------------------------------------

    def expectation(self, observable: np.ndarray) -> float:
        r"""Compute ``⟨ψ|O|ψ⟩`` for an observable ``O``.

        Parameters
        ----------
        observable : numpy.ndarray
            Hermitian operator of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        float
            Real expectation value.

        Examples
        --------
        >>> sv = Statevector([1, 0])  # |0⟩
        >>> sv.expectation(np.array([[1, 0], [0, -1]]))  # ⟨Z⟩
        1.0
        """
        O = np.asarray(observable, dtype=_COMPLEX_DTYPE)
        if O.shape != (self._dim, self._dim):
            raise ValueError(f"Observable shape {O.shape} incompatible with dim {self._dim}")
        result = np.vdot(self._data, O @ self._data)
        # Expectation of a Hermitian operator is real
        return float(np.real(result))

    # -- State properties ----------------------------------------------------

    def purity(self) -> float:
        r"""Compute purity ``Tr(ρ²)`` where ``ρ = |ψ⟩⟨ψ|``.

        Always 1.0 for a normalised state vector (up to floating point).
        """
        return float(np.real(np.sum(np.abs(self._data) ** 4)))

    def entropy(self) -> float:
        """Von Neumann entropy. Always 0 for a pure state."""
        return 0.0

    def bloch_vectors(self) -> List[np.ndarray]:
        """Compute the Bloch vector for each individual qubit.

        Traces out all other qubits and extracts ``⟨X⟩, ⟨Y⟩, ⟨Z⟩``.

        Returns
        -------
        list of numpy.ndarray
            One ``(3,)`` vector per qubit.
        """
        vectors = []
        for i in range(self._num_qubits):
            rho = self.reduced_density_matrix([i])
            x = float(np.real(np.trace(rho @ np.array([[0, 1], [1, 0]], dtype=_COMPLEX_DTYPE))))
            y = float(np.real(np.trace(rho @ np.array([[0, -1j], [1j, 0]], dtype=_COMPLEX_DTYPE))))
            z = float(np.real(np.trace(rho @ np.array([[1, 0], [0, -1]], dtype=_COMPLEX_DTYPE))))
            vectors.append(np.array([x, y, z]))
        return vectors

    # -- Conversions ----------------------------------------------------------

    def to_density_matrix(self) -> DensityMatrix:
        """Convert to density matrix: ``ρ = |ψ⟩⟨ψ|``.

        Returns
        -------
        DensityMatrix
        """
        return DensityMatrix.from_statevector(self)

    def reduced_density_matrix(
        self,
        qubits_to_keep: Sequence[int],
    ) -> np.ndarray:
        """Compute the reduced density matrix by tracing out other qubits.

        Parameters
        ----------
        qubits_to_keep : sequence of int
            Indices of qubits to keep.

        Returns
        -------
        numpy.ndarray
            Density matrix of shape ``(2**k, 2**k)``.
        """
        dm = np.outer(self._data, np.conj(self._data))
        return _partial_trace(dm, self._num_qubits, list(qubits_to_keep))

    # -- Tensor product -------------------------------------------------------

    def tensor(self, other: Statevector) -> Statevector:
        """Compute the Kronecker product ``|ψ⟩ ⊗ |φ⟩``.

        Parameters
        ----------
        other : Statevector

        Returns
        -------
        Statevector
            Combined state with ``num_qubits = self.n + other.n``.
        """
        new_data = np.kron(self._data, other._data)
        return Statevector(new_data, normalize=False, copy=False)

    def __xor__(self, other: Statevector) -> Statevector:
        """``^`` operator as shorthand for tensor product."""
        return self.tensor(other)

    # -- Linear algebra -------------------------------------------------------

    def inner(self, other: Statevector) -> complex:
        r"""Compute ``⟨self|other⟩``.

        Parameters
        ----------
        other : Statevector

        Returns
        -------
        complex
        """
        return complex(np.vdot(self._data, other._data))

    def fidelity(self, other: Statevector) -> float:
        r"""Compute ``|⟨self|other⟩|²``.

        Parameters
        ----------
        other : Statevector

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        overlap = np.vdot(self._data, other._data)
        return float(min(1.0, np.abs(overlap) ** 2))

    # -- Dunder methods -------------------------------------------------------

    def copy(self) -> Statevector:
        """Return a deep copy."""
        return Statevector(self._data.copy(), normalize=False, copy=False)

    def __repr__(self) -> str:
        return f"Statevector(num_qubits={self._num_qubits}, dim={self._dim})"

    def __str__(self) -> str:
        lines = [f"|ψ⟩ ({self._num_qubits} qubits):"]
        for i in range(self._dim):
            amp = self._data[i]
            if np.abs(amp) > _TOLERANCE:
                bits = format(i, f"0{self._num_qubits}b")
                if np.abs(amp.imag) < _TOLERANCE:
                    coeff = f"{amp.real:+.4f}"
                elif np.abs(amp.real) < _TOLERANCE:
                    coeff = f"{amp.imag:+.4f}j"
                else:
                    coeff = f"{amp:.4f}"
                lines.append(f"  {coeff}|{bits}⟩")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Statevector):
            return NotImplemented
        if self._dim != other._dim:
            return False
        # States equal up to a global phase
        # Check by comparing element-wise ratios
        nonzero = np.abs(self._data) > _TOLERANCE
        if not np.any(nonzero):
            return np.allclose(other._data, 0, atol=_TOLERANCE)
        phase = self._data[nonzero][0] / (other._data[nonzero][0] + 1e-30)
        return np.allclose(self._data, phase * other._data, atol=_TOLERANCE)

    def __len__(self) -> int:
        return self._dim

    def __getitem__(self, index: int) -> complex:
        return complex(self._data[index])


# ---------------------------------------------------------------------------
# DensityMatrix
# ---------------------------------------------------------------------------

class DensityMatrix(QuantumState):
    """General quantum state as a density matrix ``ρ``.

    Parameters
    ----------
    data : array_like
        Density matrix of shape ``(2ⁿ, 2ⁿ)``.
    num_qubits : int, optional
        Explicit number of qubits. Inferred from shape if not given.
    copy : bool, optional
        Copy the input. Default ``True``.

    Attributes
    ----------
    data : numpy.ndarray
    dim : int
    num_qubits : int

    Examples
    --------
    >>> dm = DensityMatrix.from_statevector(Statevector.zero(2))
    >>> dm.is_pure()
    True
    >>> dm.purity()
    1.0
    """

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[Sequence[complex]]],
        num_qubits: Optional[int] = None,
        copy: bool = True,
    ) -> None:
        arr = np.asarray(data, dtype=_COMPLEX_DTYPE)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {arr.shape}")

        if copy:
            arr = arr.copy()

        self._data: np.ndarray = arr
        self._dim: int = arr.shape[0]

        if num_qubits is not None:
            self._num_qubits = num_qubits
        else:
            self._num_qubits = int(round(math.log2(self._dim))) if self._dim > 0 else 0

        if (1 << self._num_qubits) != self._dim:
            raise ValueError(
                f"Dimension must be a power of 2, got {self._dim}"
            )

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def zero(cls, num_qubits: int) -> DensityMatrix:
        """Create the maximally mixed state on ``num_qubits`` qubits.

        Actually returns the |0⟩ state (pure), matching convention.
        For maximally mixed, use :meth:`maximally_mixed`.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        DensityMatrix
        """
        sv = Statevector.zero(num_qubits)
        return cls.from_statevector(sv)

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> DensityMatrix:
        """Create the maximally mixed state ``ρ = I / 2ⁿ``.

        Parameters
        ----------
        num_qubits : int

        Returns
        -------
        DensityMatrix
        """
        dim = 1 << num_qubits
        return cls(np.eye(dim, dtype=_COMPLEX_DTYPE) / dim, num_qubits=num_qubits)

    @classmethod
    def from_statevector(cls, sv: Union[Statevector, np.ndarray]) -> DensityMatrix:
        """Create density matrix from a state vector: ``ρ = |ψ⟩⟨ψ|``.

        Parameters
        ----------
        sv : Statevector or numpy.ndarray

        Returns
        -------
        DensityMatrix
        """
        if isinstance(sv, Statevector):
            data = sv._data
            n = sv.num_qubits
        else:
            data = np.asarray(sv, dtype=_COMPLEX_DTYPE)
            n = int(round(math.log2(len(data))))
        dm = np.outer(data, np.conj(data))
        return cls(dm, num_qubits=n, copy=False)

    # -- Properties ----------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """numpy.ndarray: Copy of the density matrix."""
        return self._data.copy()

    @property
    def dim(self) -> int:
        """int: Hilbert space dimension."""
        return self._dim

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits."""
        return self._num_qubits

    # -- Purity & entropy ----------------------------------------------------

    def purity(self) -> float:
        r"""Compute ``Tr(ρ²)``.

        Returns
        -------
        float
            1.0 for a pure state, < 1.0 for a mixed state.
        """
        return float(np.real(np.trace(self._data @ self._data)))

    def is_pure(self) -> bool:
        """Check whether the state is pure (purity ≈ 1)."""
        return self.purity() > 1.0 - _TOLERANCE

    def is_mixed(self) -> bool:
        """Check whether the state is mixed (purity < 1)."""
        return not self.is_pure()

    def von_neumann_entropy(self) -> float:
        r"""Compute ``S(ρ) = -Tr(ρ log₂ ρ)``.

        Returns
        -------
        float
            Non-negative entropy in bits. 0 for pure states.
        """
        eigenvalues = np.linalg.eigvalsh(self._data)
        # Clip near-zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > _TOLERANCE]
        if len(eigenvalues) == 0:
            return 0.0
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    # -- Fidelity & trace distance -------------------------------------------

    def fidelity(self, other: Union[DensityMatrix, Statevector]) -> float:
        r"""Compute the Uhlmann fidelity.

        For two density matrices:

        .. math:: F(ρ, σ) = (\\text{Tr}\\sqrt{\\sqrt{ρ} σ \\sqrt{ρ}})^2

        If ``other`` is a :class:`Statevector`, it is converted to a
        density matrix first.

        Parameters
        ----------
        other : DensityMatrix or Statevector

        Returns
        -------
        float
            Fidelity in ``[0, 1]``.
        """
        if isinstance(other, Statevector):
            other = other.to_density_matrix()

        # Use eigenvalue method for numerical stability
        sqrt_rho = _matrix_sqrt(self._data)
        product = sqrt_rho @ other._data @ sqrt_rho
        sqrt_product = _matrix_sqrt(product)
        fid = float(np.real(np.trace(sqrt_product) ** 2))
        return min(1.0, max(0.0, fid))

    def trace_distance(self, other: Union[DensityMatrix, Statevector]) -> float:
        r"""Compute the trace distance ``½ Tr|ρ - σ|``.

        Parameters
        ----------
        other : DensityMatrix or Statevector

        Returns
        -------
        float
            Non-negative distance in ``[0, 1]``.
        """
        if isinstance(other, Statevector):
            other = other.to_density_matrix()
        diff = self._data - other._data
        # |A| = sqrt(A† A)
        abs_diff = _matrix_sqrt(diff.conj().T @ diff)
        return float(0.5 * np.real(np.trace(abs_diff)))

    # -- Evolution ------------------------------------------------------------

    def evolve(self, unitary: np.ndarray) -> DensityMatrix:
        r"""Apply a unitary: ``ρ' = U ρ U†``.

        Parameters
        ----------
        unitary : numpy.ndarray
            Unitary of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        DensityMatrix
        """
        U = np.asarray(unitary, dtype=_COMPLEX_DTYPE)
        return DensityMatrix(U @ self._data @ U.conj().T, num_qubits=self._num_qubits, copy=False)

    def evolve_kraus(self, kraus_ops: Sequence[np.ndarray]) -> DensityMatrix:
        r"""Apply a general quantum operation (CPTP map) via Kraus operators.

        .. math:: ρ' = \\sum_k E_k ρ E_k^†

        Parameters
        ----------
        kraus_ops : sequence of numpy.ndarray
            Kraus operators, each of shape ``(2ⁿ, 2ⁿ)``.

        Returns
        -------
        DensityMatrix
        """
        result = np.zeros_like(self._data)
        for E in kraus_ops:
            E = np.asarray(E, dtype=_COMPLEX_DTYPE)
            result += E @ self._data @ E.conj().T
        return DensityMatrix(result, num_qubits=self._num_qubits, copy=False)

    # -- Partial trace --------------------------------------------------------

    def partial_trace(self, qubits_to_keep: Sequence[int]) -> DensityMatrix:
        """Trace out qubits not in ``qubits_to_keep``.

        Parameters
        ----------
        qubits_to_keep : sequence of int
            Qubit indices to retain.

        Returns
        -------
        DensityMatrix
            Reduced density matrix.
        """
        kept = list(qubits_to_keep)
        reduced = _partial_trace(self._data, self._num_qubits, kept)
        k = len(kept)
        return DensityMatrix(reduced, num_qubits=k, copy=False)

    # -- Tensor product -------------------------------------------------------

    def tensor_product(self, other: DensityMatrix) -> DensityMatrix:
        r"""Compute ``ρ ⊗ σ``.

        Parameters
        ----------
        other : DensityMatrix

        Returns
        -------
        DensityMatrix
        """
        new_data = np.kron(self._data, other._data)
        return DensityMatrix(new_data, num_qubits=self._num_qubits + other._num_qubits, copy=False)

    # -- Conversions ----------------------------------------------------------

    def to_density_matrix(self) -> DensityMatrix:
        """Return self (already a density matrix)."""
        return self.copy()

    def copy(self) -> DensityMatrix:
        """Return a deep copy."""
        return DensityMatrix(self._data.copy(), num_qubits=self._num_qubits, copy=False)

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        return f"DensityMatrix(num_qubits={self._num_qubits}, dim={self._dim})"

    def __str__(self) -> str:
        lines = [f"ρ ({self._num_qubits} qubits, pure={self.is_pure()})"]
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityMatrix):
            return NotImplemented
        return np.allclose(self._data, other._data, atol=_TOLERANCE)

    def __add__(self, other: DensityMatrix) -> DensityMatrix:
        if not isinstance(other, DensityMatrix):
            return NotImplemented
        combined = self._data + other._data
        # Re-normalise trace to 1
        tr = np.trace(combined)
        if tr > _TOLERANCE:
            combined /= tr
        return DensityMatrix(combined, num_qubits=self._num_qubits, copy=False)

    def __sub__(self, other: DensityMatrix) -> DensityMatrix:
        if not isinstance(other, DensityMatrix):
            return NotImplemented
        combined = self._data - other._data
        tr = np.trace(combined)
        if tr > _TOLERANCE:
            combined /= tr
        return DensityMatrix(combined, num_qubits=self._num_qubits, copy=False)

    def __mul__(self, scalar: float) -> DensityMatrix:
        """Scalar multiplication (note: does NOT preserve trace=1)."""
        return DensityMatrix(self._data * scalar, num_qubits=self._num_qubits, copy=False)

    def __rmul__(self, scalar: float) -> DensityMatrix:
        return self.__mul__(scalar)


# ---------------------------------------------------------------------------
# Operator / Observable
# ---------------------------------------------------------------------------

class Operator:
    """A linear operator on a Hilbert space.

    Not necessarily Hermitian. Stores a matrix of shape ``(2ⁿ, 2ⁿ)``.

    Parameters
    ----------
    data : array_like
        Matrix of shape ``(2ⁿ, 2ⁿ)``.
    num_qubits : int, optional
        Explicit qubit count.

    Attributes
    ----------
    data : numpy.ndarray
    dim : int
    num_qubits : int
    """

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[Sequence[complex]]],
        num_qubits: Optional[int] = None,
    ) -> None:
        arr = np.asarray(data, dtype=_COMPLEX_DTYPE)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Operator must be a square matrix")
        self._data = arr.copy()
        self._dim = arr.shape[0]
        if num_qubits is not None:
            self._num_qubits = num_qubits
        else:
            self._num_qubits = int(round(math.log2(self._dim))) if self._dim > 0 else 0

    @property
    def data(self) -> np.ndarray:
        """numpy.ndarray: Copy of the operator matrix."""
        return self._data.copy()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check whether the operator is Hermitian (self-adjoint)."""
        return np.allclose(self._data, self._data.conj().T, atol=tol)

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """Check whether the operator is unitary."""
        prod = self._data @ self._data.conj().T
        return np.allclose(prod, np.eye(self._dim), atol=tol)

    def conjugate(self) -> Operator:
        """Return the complex conjugate."""
        return Operator(self._data.conj(), self._num_qubits)

    def transpose(self) -> Operator:
        """Return the transpose."""
        return Operator(self._data.T, self._num_qubits)

    def adjoint(self) -> Operator:
        """Return the Hermitian adjoint (conjugate transpose)."""
        return Operator(self._data.conj().T, self._num_qubits)

    def compose(self, other: Operator) -> Operator:
        """Return ``self @ other`` (composition)."""
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")
        return Operator(self._data @ other._data, self._num_qubits)

    def tensor(self, other: Operator) -> Operator:
        """Return the Kronecker (tensor) product."""
        return Operator(
            np.kron(self._data, other._data),
            self._num_qubits + other._num_qubits,
        )

    def eigenvalues(self) -> np.ndarray:
        """Return eigenvalues."""
        return np.linalg.eigvalsh(self._data) if self.is_hermitian() else np.linalg.eigvals(self._data)

    def expectation(self, state: Union[Statevector, DensityMatrix]) -> float:
        r"""Compute ``⟨O⟩`` for a given state.

        For state vectors: ``⟨ψ|O|ψ⟩``.
        For density matrices: ``Tr(Oρ)``.

        Parameters
        ----------
        state : Statevector or DensityMatrix

        Returns
        -------
        float
        """
        if isinstance(state, Statevector):
            return state.expectation(self._data)
        elif isinstance(state, DensityMatrix):
            return float(np.real(np.trace(self._data @ state._data)))
        raise TypeError(f"Expected Statevector or DensityMatrix, got {type(state)}")

    def __repr__(self) -> str:
        herm = ", hermitian" if self.is_hermitian() else ""
        unit = ", unitary" if self.is_unitary() else ""
        return f"Operator(num_qubits={self._num_qubits}{herm}{unit})"

    def __matmul__(self, other: Union[Operator, np.ndarray]) -> Operator:
        if isinstance(other, Operator):
            return self.compose(other)
        return Operator(self._data @ np.asarray(other, dtype=_COMPLEX_DTYPE), self._num_qubits)


class Observable(Operator):
    """A Hermitian operator representing a physical observable.

    All observables must be Hermitian to guarantee real eigenvalues
    and real expectation values.

    Parameters
    ----------
    data : array_like
        Hermitian matrix of shape ``(2ⁿ, 2ⁿ)``.
    num_qubits : int, optional
    label : str, optional
        Descriptive name (e.g. ``'H'``, ``'Total Spin'``).
    """

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[Sequence[complex]]],
        num_qubits: Optional[int] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(data, num_qubits)
        if not self.is_hermitian():
            warnings.warn(
                "Observable should be Hermitian. "
                f"Max deviation from Hermiticity: {np.max(np.abs(self._data - self._data.conj().T))}",
                UserWarning,
                stacklevel=2,
            )
        self._label = label

    @property
    def label(self) -> Optional[str]:
        """str or None: Descriptive name."""
        return self._label

    def eigenvalues(self) -> np.ndarray:
        """Real eigenvalues (sorted ascending)."""
        return np.sort(np.linalg.eigvalsh(self._data))

    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(eigenvalues, eigenvectors)``.

        Returns
        -------
        numpy.ndarray
            Real eigenvalues, sorted ascending.
        numpy.ndarray
            Corresponding eigenvectors as columns.
        """
        return np.linalg.eigh(self._data)

    def spectral_decomposition(self) -> List[Tuple[float, np.ndarray]]:
        """Return a list of ``(eigenvalue, eigenvector)`` pairs."""
        vals, vecs = self.eigenvectors()
        return [(float(vals[i]), vecs[:, i]) for i in range(len(vals))]

    def expect(self, state: Union[Statevector, DensityMatrix]) -> float:
        """Compute expectation value (real)."""
        return self.expectation(state)

    def sample_eigenvalue(
        self,
        state: Union[Statevector, DensityMatrix],
        shots: int = 1024,
    ) -> Dict[float, int]:
        """Sample eigenvalues from the Born distribution.

        Parameters
        ----------
        state : Statevector or DensityMatrix
        shots : int

        Returns
        -------
        dict
            Mapping from eigenvalue to count.
        """
        vals, vecs = self.eigenvectors()
        if isinstance(state, Statevector):
            # Project state onto eigenbasis
            probs = np.abs(vecs.conj().T @ state._data) ** 2
        else:
            probs = np.diag(vecs.conj().T @ state._data @ vecs)
            probs = np.real(probs)
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total > _TOLERANCE:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

        indices = np.random.choice(len(vals), size=shots, p=probs)
        counts: Dict[float, int] = {}
        for idx in indices:
            v = float(vals[idx])
            # Round to avoid floating-point key issues
            v_rounded = round(v, 10)
            counts[v_rounded] = counts.get(v_rounded, 0) + 1
        return counts

    def __repr__(self) -> str:
        label = f", label={self._label!r}" if self._label else ""
        return f"Observable(num_qubits={self._num_qubits}{label})"


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _partial_trace(
    rho: np.ndarray,
    num_qubits: int,
    qubits_to_keep: Sequence[int],
) -> np.ndarray:
    """Trace out qubits not in ``qubits_to_keep`` from density matrix ``rho``.

    Uses the reshape-and-sum method.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix of shape ``(2**n, 2**n)``.
    num_qubits : int
        Total number of qubits ``n``.
    qubits_to_keep : sequence of int
        Indices of qubits to retain.

    Returns
    -------
    numpy.ndarray
        Reduced density matrix of shape ``(2**k, 2**k)``.
    """
    n = num_qubits
    k = len(qubits_to_keep)
    if k == n:
        return rho.copy()
    if k == 0:
        return np.array([[1.0 + 0j]])

    kept = list(qubits_to_keep)
    traced = [i for i in range(n) if i not in kept]

    # Reshape rho as a tensor with indices [row_0, row_1, ..., col_0, col_1, ...]
    rho_tensor = rho.reshape([2] * (2 * n))

    # Build the contraction: keep row and column axes for kept qubits
    # and sum (trace) over axes for traced qubits
    row_keep = kept
    col_keep = [q + n for q in kept]
    row_trace = traced
    col_trace = [q + n for q in traced]

    # Full list of axes
    all_axes = list(range(2 * n))

    # Result axes: row_keep + col_keep
    result_axes = row_keep + col_keep

    # Contract over (row_trace[i], col_trace[i]) pairs
    rho_reduced = rho_tensor.copy()
    for rt, ct in zip(row_trace, col_trace):
        # Sum over diagonal: keep only elements where row_trace[i] == col_trace[i]
        new_shape = []
        axes_to_sum = []
        for ax in range(rho_reduced.ndim):
            if ax == rt or ax == ct:
                axes_to_sum.append(ax)
            else:
                new_shape.append(rho_reduced.shape[ax])

        # Manual partial trace over one pair of axes
        dim_trace = rho_reduced.shape[rt]
        reduced = np.zeros(new_shape, dtype=_COMPLEX_DTYPE)
        for idx in range(dim_trace):
            # Select slice where rt==idx and ct==idx
            slicing = [slice(None)] * rho_reduced.ndim
            slicing[rt] = idx
            slicing[ct] = idx
            reduced += rho_reduced[tuple(slicing)]
        rho_reduced = reduced

    return rho_reduced.reshape((1 << k, 1 << k))


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute the principal square root of a matrix via eigendecomposition.

    Parameters
    ----------
    A : numpy.ndarray
        Positive semi-definite matrix.

    Returns
    -------
    numpy.ndarray
        ``√A`` such that ``√A @ √A ≈ A``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Clamp negative eigenvalues to 0 (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_diag = np.diag(np.sqrt(eigenvalues))
    return eigenvectors @ sqrt_diag @ eigenvectors.conj().T
