"""
Quantum Gate Definitions — 50+ Gates
======================================

Comprehensive library of quantum gates for the QuantumFlow framework.
All gates inherit from :class:`Gate` and compute their unitary matrices
lazily via ``@functools.cached_property``.

Gate categories
---------------
**Single-qubit gates**: H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, Phase,
U1, U2, U3, U, RX, RY, RZ, Rot, GlobalPhase.

**Two-qubit gates**: CNOT (CX), CZ, SWAP, iSWAP, ECR, XX (RXX), YY (RYY),
ZZ (RZZ), XY, CRX, CRY, CRZ, CPhase, CSwap (Fredkin), DCX, MS.

**Three-qubit gates**: Toffoli (CCX), Fredkin (CCZ — controlled-controlled Z),
CCZ.

**Multi-qubit gates**: MultiControlledX (MCX), MultiControlledZ (MCZ).

**Special**: Measurement (non-unitary), CompositeGate (sequence).

Helper functions
----------------
* :func:`controlled(gate, n_controls)` — add control qubits.
* :func:`power(gate, exponent)` — raise a gate to a power.
* :func:`dagger(gate)` — Hermitian adjoint (inverse).

Typical usage::

    >>> from quantumflow.core.gate import HGate, CNOTGate, RXGate
    >>> h = HGate()
    >>> h.matrix   # cached 2×2 unitary
    >>> cx = CNOTGate()
    >>> cx.to_matrix()
    >>> rx = RXGate(theta=0.5)
    >>> rx.to_matrix(0.5)

    >>> from quantumflow.core.gate import GateLibrary
    >>> lib = GateLibrary()
    >>> lib['h']    # HGate instance
    >>> lib['cx']   # CNOTGate instance
"""

from __future__ import annotations

import functools
import math
import copy
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

__all__ = [
    # Base classes
    "Gate",
    "UnitaryGate",
    "ParameterizedGate",
    "ControlledGate",
    "CompositeGate",
    "Measurement",
    # Single-qubit
    "HGate", "XGate", "YGate", "ZGate",
    "SGate", "SdgGate", "TGate", "TdgGate",
    "SXGate", "SXdgGate",
    "PhaseGate", "U1Gate", "U2Gate", "U3Gate", "UGate",
    "RXGate", "RYGate", "RZGate", "RotGate",
    "GlobalPhaseGate",
    # Two-qubit
    "CNOTGate", "CXGate", "CZGate",
    "SwapGate", "ISwapGate", "ESCGate",
    "RXXGate", "RYYGate", "RZZGate", "RZXGate", "XYGate",
    "CRXGate", "CRYGate", "CRZGate", "CPhaseGate",
    "CSwapGate", "DCXGate", "MSGate",
    # Three-qubit
    "ToffoliGate", "CCXGate", "FredkinGate", "CCZGate",
    # Multi-qubit
    "MCXGate", "MCZGate",
    # Helpers
    "controlled",
    "power",
    "dagger",
    "GateLibrary",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT2_INV = 1.0 / math.sqrt(2.0)
_PI = math.pi
_I = 1j

# Standard 2×2 matrices (used across many gates)
_H = np.array([[_SQRT2_INV, _SQRT2_INV],
               [_SQRT2_INV, -_SQRT2_INV]], dtype=np.complex128)

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -_I], [_I, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)

_S = np.array([[1, 0], [0, _I]], dtype=np.complex128)
_S_DG = np.array([[1, 0], [0, -_I]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(_I * _PI / 4)]], dtype=np.complex128)
_T_DG = np.array([[1, 0], [0, np.exp(-_I * _PI / 4)]], dtype=np.complex128)

_SX = np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                 [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex128)
_SX_DG = np.array([[0.5 - 0.5j, 0.5 + 0.5j],
                     [0.5 + 0.5j, 0.5 - 0.5j]], dtype=np.complex128)


def _is_unitary(m: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether *m* is unitary up to tolerance."""
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    prod = m @ m.conj().T
    return np.allclose(prod, np.eye(m.shape[0]), atol=atol)


# ---------------------------------------------------------------------------
# Gate base class
# ---------------------------------------------------------------------------

_G = TypeVar("_G", bound="Gate")


class Gate(ABC):
    """Abstract base class for all quantum gates.

    Every concrete gate subclass must set :attr:`name`, :attr:`num_qubits`,
    and :attr:`num_params` and implement :meth:`_compute_matrix`.

    The unitary matrix is computed lazily on first access and cached
    via ``@functools.cached_property``.

    Parameters
    ----------
    label : str, optional
        Override the display label for this gate instance.

    Attributes
    ----------
    name : str
        Short name (e.g. ``'h'``, ``'cx'``, ``'rx'``).
    num_qubits : int
        Number of qubits the gate acts on.
    num_params : int
        Number of real parameters (0 for fixed gates).
    label : str or None
        Custom label, shown in circuit diagrams.

    Examples
    --------
    Subclassing::

        class MyGate(Gate):
            name = "my"
            num_qubits = 1
            num_params = 0

            @functools.cached_property
            def matrix(self) -> np.ndarray:
                return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    """

    name: str = ""
    num_qubits: int = 0
    num_params: int = 0

    def __init__(self, label: Optional[str] = None) -> None:
        self._label = label

    @property
    def label(self) -> Optional[str]:
        """str or None: Custom display label."""
        return self._label

    @label.setter
    def label(self, value: Optional[str]) -> None:
        self._label = value

    # -- Matrix computation --------------------------------------------------

    @functools.cached_property
    def matrix(self) -> np.ndarray:
        """numpy.ndarray: Unitary matrix of shape ``(2**n, 2**n)``.

        This is computed once and cached. Subclasses must implement
        :meth:`_compute_matrix`.
        """
        m = self._compute_matrix()
        # Validate for small matrices
        if m.shape[0] <= 64:
            assert _is_unitary(m), f"{self.name} produced non-unitary matrix"
        return m

    @abstractmethod
    def _compute_matrix(self) -> np.ndarray:
        """Compute and return the gate's unitary matrix.

        Subclasses must override this.

        Returns
        -------
        numpy.ndarray
        """
        ...

    def to_matrix(self, *params: float) -> np.ndarray:
        """Return the unitary matrix, optionally with parameters.

        For parameterised gates this re-computes the matrix with the
        given angles.  For non-parameterised gates it returns the cached
        matrix.

        Parameters
        ----------
        *params : float
            Parameter values. Length must equal :attr:`num_params`.

        Returns
        -------
        numpy.ndarray
        """
        if self.num_params == 0:
            return self.matrix
        if len(params) != self.num_params:
            raise ValueError(
                f"{self.name} expects {self.num_params} params, got {len(params)}"
            )
        return self._matrix_with_params(*params)

    def _matrix_with_params(self, *params: float) -> np.ndarray:
        """Compute the matrix for specific parameter values.

        Default implementation computes from scratch. Subclasses with
        parameters should override for efficiency.
        """
        # For parameterised gates that override this, the default is to
        # build a new temporary instance, bind params, and compute.
        gate = copy.copy(self)
        gate._params = params  # type: ignore[attr-defined]
        # Bypass cache
        return gate._compute_matrix()

    # -- Inverse / dagger / controlled ----------------------------------------

    def inverse(self) -> Gate:
        """Return the inverse (Hermitian conjugate) of this gate.

        Returns
        -------
        Gate
            A new gate instance whose matrix is ``U†``.
        """
        return UnitaryGate(
            self.matrix.conj().T,
            name=f"{self.name}†",
            num_qubits=self.num_qubits,
            label=self._label,
        )

    def controlled(
        self,
        n_controls: int = 1,
    ) -> ControlledGate:
        """Wrap this gate with ``n_controls`` control qubits.

        Parameters
        ----------
        n_controls : int, optional
            Number of control qubits (default 1).

        Returns
        -------
        ControlledGate
        """
        return ControlledGate(self, n_controls=n_controls)

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        label = f", label={self._label!r}" if self._label else ""
        return f"{self.__class__.__name__}({label})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Gate):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return self._label == other._label

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._label))

    def __copy__(self) -> Gate:
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        # Clear cached matrix so it's recomputed if needed
        if "matrix" in new.__dict__:
            del new.__dict__["matrix"]
        return new

    def copy(self) -> Gate:
        """Return a shallow copy of this gate."""
        return self.__copy__()


# ---------------------------------------------------------------------------
# UnitaryGate — arbitrary unitary
# ---------------------------------------------------------------------------

class UnitaryGate(Gate):
    """A gate defined by an arbitrary unitary matrix.

    Parameters
    ----------
    matrix : array_like
        Unitary matrix of shape ``(2**n, 2**n)``.
    name : str, optional
        Gate name.
    num_qubits : int, optional
        Number of qubits. Inferred from matrix shape if not given.
    label : str, optional
        Display label.

    Examples
    --------
    >>> U = UnitaryGate([[0, 1], [1, 0]], name='my_x')
    >>> U.num_qubits
    1
    """

    def __init__(
        self,
        matrix: Union[np.ndarray, Sequence, Sequence[Sequence]],
        name: str = "unitary",
        num_qubits: Optional[int] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label=label)
        m = np.asarray(matrix, dtype=np.complex128)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Matrix must be square")
        self._matrix_data: np.ndarray = m
        self.name = name
        self.num_qubits = num_qubits if num_qubits is not None else int(round(math.log2(m.shape[0])))
        self.num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return self._matrix_data.copy()

    def inverse(self) -> UnitaryGate:
        return UnitaryGate(
            self._matrix_data.conj().T,
            name=f"{self.name}†",
            num_qubits=self.num_qubits,
            label=self._label,
        )


# ---------------------------------------------------------------------------
# ParameterizedGate
# ---------------------------------------------------------------------------

class ParameterizedGate(Gate):
    """Base for gates that take real parameters (rotation angles etc.).

    Subclasses should override :meth:`_compute_matrix` and access the
    parameters stored in :attr:`_params`.

    Parameters
    ----------
    params : tuple of float
        Parameter values.
    label : str, optional
    """

    num_params: int = 1  # default; subclasses should override

    def __init__(
        self,
        params: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label=label)
        self._params: Tuple[float, ...] = tuple(params) if params is not None else ()

    @property
    def params(self) -> Tuple[float, ...]:
        """tuple of float: Current parameter values."""
        return self._params

    def _compute_matrix(self) -> np.ndarray:
        """Subclasses must implement this using ``self._params``."""
        raise NotImplementedError

    def to_matrix(self, *params: float) -> np.ndarray:
        if len(params) != self.num_params:
            raise ValueError(
                f"{self.name} expects {self.num_params} params, got {len(params)}"
            )
        old = self._params
        self._params = tuple(params)
        try:
            # Bypass cache and recompute
            m = self._compute_matrix()
        finally:
            self._params = old
        return m


# ---------------------------------------------------------------------------
# ControlledGate
# ---------------------------------------------------------------------------

class ControlledGate(Gate):
    """Wrap an arbitrary gate with one or more control qubits.

    The resulting unitary acts on ``n_controls + base_gate.num_qubits``
    qubits.

    Parameters
    ----------
    base_gate : Gate
        The gate to control.
    n_controls : int, optional
        Number of control qubits. Default is 1.

    Examples
    --------
    >>> from quantumflow.core.gate import XGate
    >>> cx = ControlledGate(XGate(), n_controls=1)
    >>> cx.num_qubits
    2
    >>> ccx = ControlledGate(XGate(), n_controls=2)
    >>> ccx.num_qubits
    3
    """

    def __init__(self, base_gate: Gate, n_controls: int = 1) -> None:
        super().__init__()
        self._base = base_gate
        self._n_controls = n_controls
        self.name = f"c{'c' * max(n_controls - 1, 0)}{base_gate.name}"
        self.num_qubits = n_controls + base_gate.num_qubits
        self.num_params = base_gate.num_params

    @property
    def base_gate(self) -> Gate:
        """Gate: The controlled gate."""
        return self._base

    @property
    def n_controls(self) -> int:
        """int: Number of control qubits."""
        return self._n_controls

    def _compute_matrix(self) -> np.ndarray:
        n = self.num_qubits
        dim = 1 << n
        U = np.eye(dim, dtype=np.complex128)

        base_mat = self._base.matrix
        base_dim = base_mat.shape[0]

        # Control space dimension
        ctrl_dim = 1 << self._n_controls

        # The block starts at index (ctrl_dim - 1) * base_dim
        start = (ctrl_dim - 1) * base_dim
        U[start:start + base_dim, start:start + base_dim] = base_mat
        return U

    def inverse(self) -> ControlledGate:
        return ControlledGate(self._base.inverse(), n_controls=self._n_controls)


# ---------------------------------------------------------------------------
# CompositeGate
# ---------------------------------------------------------------------------

class CompositeGate(Gate):
    """A sequence of gates treated as a single gate.

    The unitary is the product of all sub-gate unitaries (in order).

    Parameters
    ----------
    gates : sequence of Gate
        Gates to compose.
    name : str, optional
    num_qubits : int, optional
        Total number of qubits. Inferred if not given.
    """

    def __init__(
        self,
        gates: Optional[Sequence[Gate]] = None,
        name: str = "composite",
        num_qubits: Optional[int] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(label=label)
        self._gates: List[Gate] = list(gates) if gates else []
        self.name = name
        self.num_qubits = num_qubits if num_qubits is not None else (
            max((g.num_qubits for g in self._gates), default=0)
        )
        self.num_params = 0

    @property
    def gates(self) -> List[Gate]:
        """list of Gate: Constituent gates."""
        return list(self._gates)

    def _compute_matrix(self) -> np.ndarray:
        dim = 1 << self.num_qubits
        result = np.eye(dim, dtype=np.complex128)
        for g in self._gates:
            result = g.matrix @ result
        return result

    def inverse(self) -> CompositeGate:
        return CompositeGate(
            [g.inverse() for g in reversed(self._gates)],
            name=f"{self.name}†",
            num_qubits=self.num_qubits,
        )


# ---------------------------------------------------------------------------
# Measurement (not a gate)
# ---------------------------------------------------------------------------

class Measurement:
    """Classical measurement instruction (non-unitary).

    Parameters
    ----------
    num_qubits : int, optional
        Number of qubits to measure. Default 1.

    Attributes
    ----------
    name : str
    num_qubits : int
    """

    name: str = "measure"
    num_params: int = 0

    def __init__(self, num_qubits: int = 1, label: Optional[str] = None) -> None:
        self.num_qubits = num_qubits
        self._label = label

    @property
    def label(self) -> Optional[str]:
        return self._label

    def inverse(self) -> Measurement:
        raise NotImplementedError("Measurement is irreversible")

    def __repr__(self) -> str:
        label = f", label={self._label!r}" if self._label else ""
        return f"Measurement(num_qubits={self.num_qubits}{label})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measurement):
            return NotImplemented
        return self.num_qubits == other.num_qubits

    def __hash__(self) -> int:
        return hash(("measure", self.num_qubits))


# ===========================================================================
# SINGLE-QUBIT GATES
# ===========================================================================

class HGate(Gate):
    """Hadamard gate.

    .. math::

        H = \\frac{1}{\\sqrt{2}} \\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}
    """
    name = "h"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _H.copy()


class XGate(Gate):
    """Pauli-X (NOT) gate.

    .. math:: X = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}
    """
    name = "x"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _X.copy()


class YGate(Gate):
    """Pauli-Y gate.

    .. math:: Y = \\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}
    """
    name = "y"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _Y.copy()


class ZGate(Gate):
    """Pauli-Z gate.

    .. math:: Z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}
    """
    name = "z"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _Z.copy()


class SGate(Gate):
    """Phase gate S = √Z.

    .. math:: S = \\begin{pmatrix} 1 & 0 \\\\ 0 & i \\end{pmatrix}
    """
    name = "s"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _S.copy()

    def inverse(self) -> SdgGate:
        return SdgGate()


class SdgGate(Gate):
    """Adjoint phase gate S†.

    .. math:: S^\\dagger = \\begin{pmatrix} 1 & 0 \\\\ 0 & -i \\end{pmatrix}
    """
    name = "sdg"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _S_DG.copy()

    def inverse(self) -> SGate:
        return SGate()


class TGate(Gate):
    """T gate (π/8 gate).

    .. math:: T = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\pi/4} \\end{pmatrix}
    """
    name = "t"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _T.copy()

    def inverse(self) -> TdgGate:
        return TdgGate()


class TdgGate(Gate):
    """Adjoint T gate.

    .. math:: T^\\dagger = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{-i\\pi/4} \\end{pmatrix}
    """
    name = "tdg"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _T_DG.copy()

    def inverse(self) -> TGate:
        return TGate()


class SXGate(Gate):
    """Square-root of X gate.

    .. math:: \\sqrt{X} = \\frac{1}{2}\\begin{pmatrix} 1+i & 1-i \\\\ 1-i & 1+i \\end{pmatrix}
    """
    name = "sx"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _SX.copy()

    def inverse(self) -> SXdgGate:
        return SXdgGate()


class SXdgGate(Gate):
    """Adjoint square-root of X gate.

    .. math:: (\\sqrt{X})^\\dagger = \\frac{1}{2}\\begin{pmatrix} 1-i & 1+i \\\\ 1+i & 1-i \\end{pmatrix}
    """
    name = "sxdg"
    num_qubits = 1
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _SX_DG.copy()

    def inverse(self) -> SXGate:
        return SXGate()


class PhaseGate(ParameterizedGate):
    """Phase gate (U1).

    .. math:: P(\\lambda) = \\begin{pmatrix} 1 & 0 \\\\ 0 & e^{i\\lambda} \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[theta]`` — phase angle in radians.
    """
    name = "p"
    num_qubits = 1
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        return np.array(
            [[1, 0], [0, np.exp(_I * theta)]],
            dtype=np.complex128,
        )


class U1Gate(PhaseGate):
    """U1 gate — alias for PhaseGate."""
    name = "u1"


class U2Gate(ParameterizedGate):
    """U2 gate.

    .. math::

        U2(\\phi, \\lambda) = \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix} 1 & -e^{i\\lambda} \\\\ e^{i\\phi} & e^{i(\\phi+\\lambda)} \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[phi, lam]``
    """
    name = "u2"
    num_qubits = 1
    num_params = 2

    def _compute_matrix(self) -> np.ndarray:
        phi = self._params[0] if len(self._params) > 0 else 0.0
        lam = self._params[1] if len(self._params) > 1 else 0.0
        return _SQRT2_INV * np.array(
            [[1, -np.exp(_I * lam)],
             [np.exp(_I * phi), np.exp(_I * (phi + lam))]],
            dtype=np.complex128,
        )


class U3Gate(ParameterizedGate):
    """U3 gate (full single-qubit unitary).

    .. math::

        U3(\\theta, \\phi, \\lambda) =
        \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2} \\\\
            e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[theta, phi, lam]``
    """
    name = "u3"
    num_qubits = 1
    num_params = 3

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if len(self._params) > 0 else 0.0
        phi = self._params[1] if len(self._params) > 1 else 0.0
        lam = self._params[2] if len(self._params) > 2 else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -np.exp(_I * lam) * s],
             [np.exp(_I * phi) * s, np.exp(_I * (phi + lam)) * c]],
            dtype=np.complex128,
        )


class UGate(U3Gate):
    """U gate — alias for U3Gate."""
    name = "u"


class RXGate(ParameterizedGate):
    """Rotation about the X axis.

    .. math:: R_X(\\theta) = \\cos\\frac{\\theta}{2}\\,I - i\\sin\\frac{\\theta}{2}\\,X

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "rx"
    num_qubits = 1
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -_I * s], [-_I * s, c]],
            dtype=np.complex128,
        )


class RYGate(ParameterizedGate):
    """Rotation about the Y axis.

    .. math:: R_Y(\\theta) = \\cos\\frac{\\theta}{2}\\,I - i\\sin\\frac{\\theta}{2}\\,Y

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "ry"
    num_qubits = 1
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -s], [s, c]],
            dtype=np.complex128,
        )


class RZGate(ParameterizedGate):
    """Rotation about the Z axis.

    .. math:: R_Z(\\theta) = \\begin{pmatrix} e^{-i\\theta/2} & 0 \\\\ 0 & e^{i\\theta/2} \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "rz"
    num_qubits = 1
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        p = np.exp(-_I * theta / 2)
        q = np.exp(_I * theta / 2)
        return np.array([[p, 0], [0, q]], dtype=np.complex128)


class RotGate(ParameterizedGate):
    """Euler rotation Rz · Ry · Rz.

    .. math::

        \\text{Rot}(\\phi, \\theta, \\omega) = R_Z(\\omega)\\,R_Y(\\theta)\\,R_Z(\\phi)

    Parameters
    ----------
    params : sequence of float
        ``[phi, theta, omega]``
    """
    name = "rot"
    num_qubits = 1
    num_params = 3

    def _compute_matrix(self) -> np.ndarray:
        phi = self._params[0] if len(self._params) > 0 else 0.0
        theta = self._params[1] if len(self._params) > 1 else 0.0
        omega = self._params[2] if len(self._params) > 2 else 0.0
        # RZ(phi) · RY(theta) · RZ(omega)
        a = math.cos(theta / 2)
        b = math.sin(theta / 2)
        return np.array(
            [
                [np.exp(-_I * (phi + omega) / 2) * a,
                 -np.exp(_I * (omega - phi) / 2) * b],
                [np.exp(-_I * (omega - phi) / 2) * b,
                 np.exp(_I * (phi + omega) / 2) * a],
            ],
            dtype=np.complex128,
        )


class GlobalPhaseGate(ParameterizedGate):
    """Global phase gate (acts on a single qubit wire but multiplies the whole state).

    .. math:: P(\\theta) = e^{i\\theta} \\, I

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "global_phase"
    num_qubits = 1
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        return np.exp(_I * theta) * _I2.copy()

    def inverse(self) -> GlobalPhaseGate:
        if self._params:
            return GlobalPhaseGate(params=[-self._params[0]])
        return GlobalPhaseGate()


# ===========================================================================
# TWO-QUBIT GATES
# ===========================================================================

class CNOTGate(Gate):
    """Controlled-NOT (controlled-X) gate.

    .. math::

        \\text{CNOT} = |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes X
    """
    name = "cx"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]],
            dtype=np.complex128,
        )


class CXGate(CNOTGate):
    """Alias for CNOTGate."""
    name = "cx"


class CZGate(Gate):
    """Controlled-Z gate.

    .. math::

        \\text{CZ} = |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes Z
    """
    name = "cz"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, -1]],
            dtype=np.complex128,
        )


class SwapGate(Gate):
    """SWAP gate.

    .. math:: \\text{SWAP} |a, b\\rangle = |b, a\\rangle
    """
    name = "swap"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=np.complex128,
        )

    def inverse(self) -> SwapGate:
        return SwapGate(label=self._label)


class ISwapGate(Gate):
    """iSWAP gate.

    .. math::

        \\text{iSWAP} = \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & i & 0 \\\\
        0 & i & 0 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{pmatrix}
    """
    name = "iswap"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 0, _I, 0],
             [0, _I, 0, 0],
             [0, 0, 0, 1]],
            dtype=np.complex128,
        )

    def inverse(self) -> ISwapGate:
        # iSWAP† has -i in the off-diagonal
        return UnitaryGate(
            np.array(
                [[1, 0, 0, 0],
                 [0, 0, -_I, 0],
                 [0, -_I, 0, 0],
                 [0, 0, 0, 1]],
                dtype=np.complex128,
            ),
            name="iswap†",
            num_qubits=2,
        )


class ESCGate(Gate):
    """Echoed Cross-Resonance (ECR) gate.

    Used in IBM's native gate set. Defined as:

    .. math::

        \\text{ECR} = (X \\otimes I) \\cdot \\text{CNOT} \\cdot (I \\otimes S) \\cdot \\text{CNOT} \\cdot (I \\otimes H)

    This produces the unitary:

    .. math::

        \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
            0 & 0 & i & i \\\\
            0 & 0 & 1 & -1 \\\\
            1 & 1 & 0 & 0 \\\\
            i & -i & 0 & 0
        \\end{pmatrix}
    """
    name = "ecr"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return _SQRT2_INV * np.array(
            [[0, 0, _I, _I],
             [0, 0, 1, -1],
             [1, 1, 0, 0],
             [_I, -_I, 0, 0]],
            dtype=np.complex128,
        )


class RXXGate(ParameterizedGate):
    """XX rotation (Ising coupling) gate.

    .. math::

        R_{XX}(\\theta) = e^{-i\\theta\\, X\\otimes X / 2} =
        \\begin{pmatrix}
            \\cos\\frac{\\theta}{2} & 0 & 0 & -i\\sin\\frac{\\theta}{2} \\\\
            0 & \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} & 0 \\\\
            0 & -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2} & 0 \\\\
            -i\\sin\\frac{\\theta}{2} & 0 & 0 & \\cos\\frac{\\theta}{2}
        \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "rxx"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        a = -_I * s
        return np.array(
            [[c, 0, 0, a],
             [0, c, a, 0],
             [0, a, c, 0],
             [a, 0, 0, c]],
            dtype=np.complex128,
        )


# Alias
XXGate = RXXGate


class RYYGate(ParameterizedGate):
    """YY rotation gate.

    .. math:: R_{YY}(\\theta) = e^{-i\\theta\\, Y\\otimes Y / 2}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "ryy"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        a = -_I * s
        return np.array(
            [[c, 0, 0, a],
             [0, c, -a, 0],
             [0, -a, c, 0],
             [a, 0, 0, c]],
            dtype=np.complex128,
        )


YYGate = RYYGate


class RZZGate(ParameterizedGate):
    """ZZ rotation gate.

    .. math:: R_{ZZ}(\\theta) = e^{-i\\theta\\, Z\\otimes Z / 2}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "rzz"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        p = np.exp(-_I * theta / 2)
        q = np.exp(_I * theta / 2)
        return np.diag([p, q, q, p])


ZZGate = RZZGate


class RZXGate(ParameterizedGate):
    """ZX rotation gate.

    .. math:: R_{ZX}(\\theta) = e^{-i\\theta\\, Z\\otimes X / 2}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "rzx"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -_I * s, 0, 0],
             [-_I * s, c, 0, 0],
             [0, 0, c, _I * s],
             [0, 0, _I * s, c]],
            dtype=np.complex128,
        )


class XYGate(ParameterizedGate):
    """XY interaction gate (Mølmer–Sørensen–style).

    .. math::

        R_{XY}(\\theta) = e^{-i\\theta\\, (X\\otimes X + Y\\otimes Y) / 2}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "xy"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = _I * math.sin(theta / 2)
        return np.array(
            [[1, 0, 0, 0],
             [0, c, s, 0],
             [0, s, c, 0],
             [0, 0, 0, 1]],
            dtype=np.complex128,
        )


class CRXGate(ParameterizedGate):
    """Controlled-RX gate.

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "crx"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, c, -_I * s],
             [0, 0, -_I * s, c]],
            dtype=np.complex128,
        )


class CRYGate(ParameterizedGate):
    """Controlled-RY gate.

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "cry"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, c, -s],
             [0, 0, s, c]],
            dtype=np.complex128,
        )


class CRZGate(ParameterizedGate):
    """Controlled-RZ gate.

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "crz"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        p = np.exp(-_I * theta / 2)
        q = np.exp(_I * theta / 2)
        return np.diag([1, 1, p, q])


class CPhaseGate(ParameterizedGate):
    """Controlled-phase gate.

    .. math::

        \\text{CPhase}(\\lambda) = \\text{diag}(1, 1, 1, e^{i\\lambda})

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "cphase"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        return np.diag([1, 1, 1, np.exp(_I * theta)])


class CSwapGate(Gate):
    """Controlled-SWAP (Fredkin) gate.

    Swaps target qubits if and only if the control is |1⟩.
    """
    name = "cswap"
    num_qubits = 3
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]],
            dtype=np.complex128,
        )

    def inverse(self) -> CSwapGate:
        return CSwapGate(label=self._label)


class DCXGate(Gate):
    """Double-CNOT (DCX) gate.

    Applies CNOT(control, target) followed by CNOT(target, control).
    """
    name = "dcx"
    num_qubits = 2
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0]],
            dtype=np.complex128,
        )


class MSGate(ParameterizedGate):
    """Mølmer–Sørensen gate.

    .. math::

        \\text{MS}(\\theta) = \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
            1 & 0 & 0 & -i e^{i\\theta} \\\\
            0 & 1 & -i & 0 \\\\
            0 & -i & 1 & 0 \\\\
            -i e^{-i\\theta} & 0 & 0 & 1
        \\end{pmatrix}

    Parameters
    ----------
    params : sequence of float
        ``[theta]``
    """
    name = "ms"
    num_qubits = 2
    num_params = 1

    def _compute_matrix(self) -> np.ndarray:
        theta = self._params[0] if self._params else 0.0
        return _SQRT2_INV * np.array(
            [[1, 0, 0, -_I * np.exp(_I * theta)],
             [0, 1, -_I, 0],
             [0, -_I, 1, 0],
             [-_I * np.exp(-_I * theta), 0, 0, 1]],
            dtype=np.complex128,
        )


# ===========================================================================
# THREE-QUBIT GATES
# ===========================================================================

class ToffoliGate(Gate):
    """Toffoli (CCNOT / CCX) gate.

    Flips the target qubit if both controls are |1⟩.
    """
    name = "ccx"
    num_qubits = 3
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=np.complex128)
        # Swap last two rows/columns for the 111↔110 block
        m[6, 6] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        m[7, 7] = 0
        return m

    def inverse(self) -> ToffoliGate:
        return ToffoliGate(label=self._label)


class CCXGate(ToffoliGate):
    """Alias for ToffoliGate."""
    name = "ccx"


class CCZGate(Gate):
    """Controlled-Controlled-Z gate.

    Applies Z to the target if both controls are |1⟩.
    """
    name = "ccz"
    num_qubits = 3
    num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        m = np.eye(8, dtype=np.complex128)
        m[7, 7] = -1
        return m

    def inverse(self) -> CCZGate:
        return CCZGate(label=self._label)


class FredkinGate(CSwapGate):
    """Fredkin (controlled-SWAP) gate — alias for CSwapGate."""
    name = "fredkin"


# ===========================================================================
# MULTI-QUBIT GATES
# ===========================================================================

class MCXGate(Gate):
    """Multi-controlled X gate.

    Flips the target qubit when *all* control qubits are |1⟩.

    Parameters
    ----------
    num_controls : int
        Number of control qubits. Total qubits = num_controls + 1.
    """

    def __init__(self, num_controls: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label=label)
        self._nc = num_controls
        self.name = f"mcx_{num_controls}"
        self.num_qubits = num_controls + 1
        self.num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        n = self.num_qubits
        dim = 1 << n
        m = np.eye(dim, dtype=np.complex128)
        # |11...1⟩ ↔ |11...0⟩ in the last two basis states
        second_last = dim - 2
        last = dim - 1
        m[second_last, second_last] = 0
        m[second_last, last] = 1
        m[last, second_last] = 1
        m[last, last] = 0
        return m

    def inverse(self) -> MCXGate:
        return MCXGate(self._nc, label=self._label)


class MCZGate(Gate):
    """Multi-controlled Z gate.

    Applies Z to the target when *all* control qubits are |1⟩.

    Parameters
    ----------
    num_controls : int
        Number of control qubits. Total qubits = num_controls + 1.
    """

    def __init__(self, num_controls: int = 2, label: Optional[str] = None) -> None:
        super().__init__(label=label)
        self._nc = num_controls
        self.name = f"mcz_{num_controls}"
        self.num_qubits = num_controls + 1
        self.num_params = 0

    def _compute_matrix(self) -> np.ndarray:
        n = self.num_qubits
        dim = 1 << n
        m = np.eye(dim, dtype=np.complex128)
        m[dim - 1, dim - 1] = -1
        return m

    def inverse(self) -> MCZGate:
        return MCZGate(self._nc, label=self._label)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def controlled(gate: Gate, n_controls: int = 1) -> ControlledGate:
    """Return a version of ``gate`` with ``n_controls`` control qubits.

    Parameters
    ----------
    gate : Gate
        The gate to control.
    n_controls : int, optional
        Number of control qubits. Default is 1.

    Returns
    -------
    ControlledGate

    Examples
    --------
    >>> from quantumflow.core.gate import XGate, controlled
    >>> c2x = controlled(XGate(), n_controls=2)
    >>> c2x.num_qubits
    3
    """
    if not isinstance(gate, Gate):
        raise TypeError(f"Expected a Gate, got {type(gate)}")
    return ControlledGate(gate, n_controls=n_controls)


def power(gate: Gate, exponent: float) -> Gate:
    """Raise a gate to a scalar power.

    Computes the matrix power ``U^exponent`` via eigendecomposition.

    Parameters
    ----------
    gate : Gate
    exponent : float

    Returns
    -------
    UnitaryGate
        New gate with matrix ``gate.matrix ** exponent``.

    Examples
    --------
    >>> from quantumflow.core.gate import XGate, power
    >>> sqrt_x = power(XGate(), 0.5)
    >>> np.allclose(sqrt_x.matrix @ sqrt_x.matrix, XGate().matrix)
    True
    """
    if not isinstance(gate, Gate):
        raise TypeError(f"Expected a Gate, got {type(gate)}")
    m = gate.matrix
    # Eigendecomposition: U = V D V†, then U^p = V D^p V†
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    # Cast to complex before raising to power to handle negative eigenvalues
    eigenvalues_c = eigenvalues.astype(np.complex128)
    # Principal branch: z^p = exp(p * log(z)), handle near-zero carefully
    # Map eigenvalues to complex unit circle for phase gates
    phases = np.exp(1j * np.angle(eigenvalues_c))
    powered_eigenvalues = phases ** exponent
    # Preserve magnitudes (eigenvalues of unitary have |λ|=1)
    result = eigenvectors @ np.diag(powered_eigenvalues) @ eigenvectors.conj().T
    return UnitaryGate(
        result,
        name=f"{gate.name}^{exponent}",
        num_qubits=gate.num_qubits,
    )


def dagger(gate: Gate) -> Gate:
    """Return the Hermitian adjoint (dagger / inverse) of a gate.

    Equivalent to :meth:`Gate.inverse`.

    Parameters
    ----------
    gate : Gate

    Returns
    -------
    Gate

    Examples
    --------
    >>> from quantumflow.core.gate import XGate, dagger
    >>> dx = dagger(XGate())
    >>> np.allclose(dx.matrix @ XGate().matrix, np.eye(2))
    True
    """
    if not isinstance(gate, Gate):
        raise TypeError(f"Expected a Gate, got {type(gate)}")
    return gate.inverse()


# ===========================================================================
# GATE LIBRARY (REGISTRY)
# ===========================================================================

class GateLibrary:
    """Registry that maps short string names to gate classes or instances.

    The library is pre-populated with all built-in gates. Users can
    register custom gates at runtime.

    Examples
    --------
    >>> lib = GateLibrary()
    >>> lib['h']    # HGate instance
    >>> lib['cx']   # CNOTGate instance
    >>> lib['rx']   # RXGate class (needs params)
    >>> lib.register('custom', MyGate())
    >>> lib['custom']
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Union[Type[Gate], Gate]] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        """Pre-populate with all built-in gates."""
        # Single-qubit gates (no params)
        no_param_gates: List[Gate] = [
            HGate(), XGate(), YGate(), ZGate(),
            SGate(), SdgGate(), TGate(), TdgGate(),
            SXGate(), SXdgGate(),
        ]
        # Single-qubit parameterised gates (store class for late binding)
        param_gate_classes: Dict[str, Type[ParameterizedGate]] = {
            "p": PhaseGate,
            "u1": U1Gate,
            "u2": U2Gate,
            "u3": U3Gate,
            "u": UGate,
            "rx": RXGate,
            "ry": RYGate,
            "rz": RZGate,
            "rot": RotGate,
            "global_phase": GlobalPhaseGate,
        }
        # Two-qubit gates (no params)
        two_qubit_gates: List[Gate] = [
            CNOTGate(), CZGate(), SwapGate(), ISwapGate(),
            ESCGate(), DCXGate(),
        ]
        # Two-qubit parameterised
        two_param_gate_classes: Dict[str, Type[ParameterizedGate]] = {
            "rxx": RXXGate,
            "ryy": RYYGate,
            "rzz": RZZGate,
            "rzx": RZXGate,
            "xy": XYGate,
            "crx": CRXGate,
            "cry": CRYGate,
            "crz": CRZGate,
            "cphase": CPhaseGate,
            "ms": MSGate,
        }
        # Three-qubit gates
        three_qubit_gates: List[Gate] = [
            ToffoliGate(), CCZGate(), CSwapGate(),
        ]

        # Register instances
        for g in no_param_gates:
            self._registry[g.name] = g

        for g in two_qubit_gates:
            self._registry[g.name] = g

        # Aliases
        self._registry["cx"] = CNOTGate()
        self._registry["fredkin"] = CSwapGate()
        self._registry["ccx"] = ToffoliGate()

        for g in three_qubit_gates:
            self._registry[g.name] = g

        # Register classes
        self._registry.update(param_gate_classes)
        self._registry.update(two_param_gate_classes)

    def register(self, name: str, gate: Union[Gate, Type[Gate]]) -> None:
        """Register a gate (instance or class) under ``name``.

        Parameters
        ----------
        name : str
            Short identifier (e.g. ``'h'``, ``'rx'``).
        gate : Gate or type
            Gate instance or Gate subclass.
        """
        self._registry[name.lower()] = gate

    def get(self, name: str) -> Optional[Union[Gate, Type[Gate]]]:
        """Look up a gate by name. Returns ``None`` if not found."""
        return self._registry.get(name.lower())

    def __getitem__(self, name: str) -> Union[Gate, Type[Gate]]:
        """Look up a gate. Raises ``KeyError`` if not found."""
        key = name.lower()
        if key not in self._registry:
            raise KeyError(
                f"Gate '{name}' not found. "
                f"Available: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry

    def __repr__(self) -> str:
        return f"GateLibrary({len(self._registry)} gates)"

    def __str__(self) -> str:
        lines = ["Gate Library:"]
        for name in sorted(self._registry):
            entry = self._registry[name]
            if isinstance(entry, type):
                lines.append(f"  {name:20s} — {entry.__name__} (parameterised)")
            else:
                lines.append(
                    f"  {name:20s} — {entry.__class__.__name__} "
                    f"({entry.num_qubits} qubits)"
                )
        return "\n".join(lines)

    def list_gates(self, category: Optional[str] = None) -> List[str]:
        """Return sorted list of gate names, optionally filtered by category.

        Parameters
        ----------
        category : str, optional
            One of ``'single'``, ``'two'``, ``'three'``, ``'multi'``,
            ``'parameterized'``, or ``None`` for all.
        """
        if category is None:
            return sorted(self._registry.keys())

        cat = category.lower()
        result = []
        for name, entry in self._registry.items():
            if isinstance(entry, type):
                if cat == "parameterized":
                    result.append(name)
                continue
            g = entry
            if cat == "single" and g.num_qubits == 1:
                result.append(name)
            elif cat == "two" and g.num_qubits == 2:
                result.append(name)
            elif cat == "three" and g.num_qubits == 3:
                result.append(name)
            elif cat == "multi" and g.num_qubits > 3:
                result.append(name)
            elif cat == "parameterized" and g.num_params > 0:
                result.append(name)
        return sorted(result)
