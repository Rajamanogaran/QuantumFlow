"""
Quantum Circuit
===============

The :class:`QuantumCircuit` is the central data structure of QuantumFlow —
it holds a sequence of quantum operations (gates, measurements, barriers,
resets) applied to named registers of qubits and classical bits.

A circuit can be constructed imperatively::

    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.rx(0.5, 2)
    >>> qc.measure([0, 1, 2], [0, 1, 2])

or declaratively from registers::

    >>> qr = QuantumRegister(2, 'q')
    >>> cr = ClassicalRegister(2, 'c')
    >>> qc = QuantumCircuit(qr, cr)

Circuits support composition, inversion, parameter binding, and export
to OpenQASM 2.0.
"""

from __future__ import annotations

import copy
import math
import random
import warnings
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

from quantumflow.core.gate import (
    CNOTGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSwapGate,
    CZGate,
    DCXGate,
    ESCGate,
    Gate,
    GlobalPhaseGate,
    HGate,
    ISwapGate,
    MCXGate,
    MCZGate,
    Measurement,
    MSGate,
    ParameterizedGate,
    PhaseGate,
    RXXGate,
    RXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    RZXGate,
    RotGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    SwapGate,
    TGate,
    TdgGate,
    ToffoliGate,
    U2Gate,
    U3Gate,
    UGate,
    UnitaryGate,
    XGate,
    XYGate,
    YGate,
    ZGate,
    CCZGate,
    CompositeGate,
    CPhaseGate,
    FredkinGate,
)
from quantumflow.core.operation import (
    Barrier,
    Operation,
    Reset,
)
from quantumflow.core.register import (
    ClassicalRegister,
    QuantumRegister,
    Register,
)
from quantumflow.core.state import Statevector

__all__ = [
    "QuantumCircuit",
]


# ---------------------------------------------------------------------------
# Parameter placeholder
# ---------------------------------------------------------------------------

class Parameter:
    """A symbolic parameter for parameterised circuits.

    Thin wrapper around a string name. Can be bound to a float later
    via :meth:`QuantumCircuit.bind_parameters`.

    Parameters
    ----------
    name : str
        Parameter identifier.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parameter):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """A quantum circuit with quantum and classical registers.

    Parameters
    ----------
    *regs : int or QuantumRegister or ClassicalRegister
        - ``int`` → creates a single :class:`QuantumRegister` of that size.
        - One or more register objects.
    name : str, optional
        Circuit name.
    metadata : dict, optional
        Arbitrary metadata.

    Attributes
    ----------
    name : str
    num_qubits : int
    num_clbits : int
    data : list of Operation

    Examples
    --------
    >>> qc = QuantumCircuit(2, name='bell')
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.measure([0, 1], [0, 1])
    >>> print(qc)
    """

    def __init__(
        self,
        *regs: Union[int, QuantumRegister, ClassicalRegister],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name: str = name if name is not None else ""
        self._metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        self._data: List[Operation] = []
        self._qregs: List[QuantumRegister] = []
        self._cregs: List[ClassicalRegister] = []
        self._qreg_map: Dict[str, Tuple[int, int]] = {}  # name -> (start, end)
        self._creg_map: Dict[str, Tuple[int, int]] = {}
        self._parameters: Dict[str, Union[float, Parameter]] = {}

        # Parse constructor arguments
        if not regs:
            # No registers — empty circuit
            pass
        else:
            # Check if first arg is an int (shorthand for QuantumRegister)
            if isinstance(regs[0], int):
                n = regs[0]
                qr = QuantumRegister(n)
                self.add_register(qr)
                # Remaining args
                for r in regs[1:]:
                    self._add_reg(r)
            else:
                for r in regs:
                    self._add_reg(r)

    # -- Register management -------------------------------------------------

    def _add_reg(self, reg: Union[QuantumRegister, ClassicalRegister]) -> None:
        """Add a single register."""
        if isinstance(reg, QuantumRegister):
            self.add_register(reg)
        elif isinstance(reg, ClassicalRegister):
            self.add_register(reg)
        else:
            raise TypeError(f"Expected QuantumRegister or ClassicalRegister, got {type(reg)}")

    def add_register(self, reg: Union[QuantumRegister, ClassicalRegister]) -> None:
        """Add a quantum or classical register to the circuit.

        Parameters
        ----------
        reg : QuantumRegister or ClassicalRegister
        """
        if isinstance(reg, QuantumRegister):
            start = self.num_qubits
            self._qregs.append(reg)
            self._qreg_map[reg.name] = (start, start + reg.size)
        elif isinstance(reg, ClassicalRegister):
            start = self.num_clbits
            self._cregs.append(reg)
            self._creg_map[reg.name] = (start, start + reg.size)
        else:
            raise TypeError(f"Expected register, got {type(reg)}")

    @property
    def qregs(self) -> List[QuantumRegister]:
        """list of QuantumRegister: Quantum registers."""
        return list(self._qregs)

    @property
    def cregs(self) -> List[ClassicalRegister]:
        """list of ClassicalRegister: Classical registers."""
        return list(self._cregs)

    @property
    def num_qubits(self) -> int:
        """int: Total number of qubits across all quantum registers."""
        return sum(r.size for r in self._qregs)

    @property
    def num_clbits(self) -> int:
        """int: Total number of classical bits."""
        return sum(r.size for r in self._cregs)

    @property
    def width(self) -> int:
        """int: Alias for ``num_qubits``."""
        return self.num_qubits

    @property
    def data(self) -> List[Operation]:
        """list of Operation: Circuit operations (read-only copy)."""
        return list(self._data)

    # -- Gate convenience methods --------------------------------------------

    def h(self, qubit: int) -> QuantumCircuit:
        """Apply Hadamard gate."""
        self.append(HGate(), [qubit])
        return self

    def x(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-X gate."""
        self.append(XGate(), [qubit])
        return self

    def y(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-Y gate."""
        self.append(YGate(), [qubit])
        return self

    def z(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-Z gate."""
        self.append(ZGate(), [qubit])
        return self

    def s(self, qubit: int) -> QuantumCircuit:
        """Apply S (phase √Z) gate."""
        self.append(SGate(), [qubit])
        return self

    def sdg(self, qubit: int) -> QuantumCircuit:
        """Apply S† gate."""
        self.append(SdgGate(), [qubit])
        return self

    def t(self, qubit: int) -> QuantumCircuit:
        """Apply T gate."""
        self.append(TGate(), [qubit])
        return self

    def tdg(self, qubit: int) -> QuantumCircuit:
        """Apply T† gate."""
        self.append(TdgGate(), [qubit])
        return self

    def sx(self, qubit: int) -> QuantumCircuit:
        """Apply √X gate."""
        self.append(SXGate(), [qubit])
        return self

    def sxdg(self, qubit: int) -> QuantumCircuit:
        """Apply (√X)† gate."""
        self.append(SXdgGate(), [qubit])
        return self

    def p(self, theta: float, qubit: int) -> QuantumCircuit:
        """Apply phase gate P(θ)."""
        self.append(PhaseGate(params=[theta]), [qubit], [theta])
        return self

    def u1(self, theta: float, qubit: int) -> QuantumCircuit:
        """Apply U1 gate (alias for phase)."""
        return self.p(theta, qubit)

    def u2(self, phi: float, lam: float, qubit: int) -> QuantumCircuit:
        """Apply U2 gate."""
        self.append(U2Gate(params=[phi, lam]), [qubit], [phi, lam])
        return self

    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> QuantumCircuit:
        """Apply U3 gate."""
        self.append(U3Gate(params=[theta, phi, lam]), [qubit], [theta, phi, lam])
        return self

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> QuantumCircuit:
        """Apply U gate (alias for U3)."""
        return self.u3(theta, phi, lam, qubit)

    def rx(self, theta: float, qubit: int) -> QuantumCircuit:
        """Apply RX(θ) rotation."""
        self.append(RXGate(params=[theta]), [qubit], [theta])
        return self

    def ry(self, theta: float, qubit: int) -> QuantumCircuit:
        """Apply RY(θ) rotation."""
        self.append(RYGate(params=[theta]), [qubit], [theta])
        return self

    def rz(self, theta: float, qubit: int) -> QuantumCircuit:
        """Apply RZ(θ) rotation."""
        self.append(RZGate(params=[theta]), [qubit], [theta])
        return self

    def rot(self, phi: float, theta: float, omega: float, qubit: int) -> QuantumCircuit:
        """Apply Rot(φ, θ, ω) = Rz(ω)·Ry(θ)·Rz(φ)."""
        self.append(RotGate(params=[phi, theta, omega]), [qubit], [phi, theta, omega])
        return self

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Apply CNOT (CX) gate."""
        self.append(CNOTGate(), [control, target])
        return self

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """Alias for :meth:`cx`."""
        return self.cx(control, target)

    def cy(self, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-Y gate."""
        from quantumflow.core.gate import ControlledGate, YGate
        self.append(ControlledGate(YGate()), [control, target])
        return self

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-Z gate."""
        self.append(CZGate(), [control, target])
        return self

    def swap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply SWAP gate."""
        self.append(SwapGate(), [qubit1, qubit2])
        return self

    def iswap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply iSWAP gate."""
        self.append(ISwapGate(), [qubit1, qubit2])
        return self

    def ecr(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply ECR gate."""
        self.append(ESCGate(), [qubit1, qubit2])  # ECR gate
        return self

    def rxx(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply RXX(θ) gate."""
        self.append(RXXGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    def ryy(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply RYY(θ) gate."""
        self.append(RYYGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    def rzz(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply RZZ(θ) gate."""
        self.append(RZZGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    def rzx(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply RZX(θ) gate."""
        self.append(RZXGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    def xy(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply XY(θ) gate."""
        self.append(XYGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    def crx(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-RX gate."""
        self.append(CRXGate(params=[theta]), [control, target], [theta])
        return self

    def cry(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-RY gate."""
        self.append(CRYGate(params=[theta]), [control, target], [theta])
        return self

    def crz(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-RZ gate."""
        self.append(CRZGate(params=[theta]), [control, target], [theta])
        return self

    def cphase(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-phase gate."""
        self.append(CPhaseGate(params=[theta]), [control, target], [theta])
        return self

    def cp(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Alias for :meth:`cphase`."""
        return self.cphase(theta, control, target)

    def ccx(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Apply Toffoli (CCX) gate."""
        self.append(ToffoliGate(), [control1, control2, target])
        return self

    def toffoli(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Alias for :meth:`ccx`."""
        return self.ccx(control1, control2, target)

    def cswap(self, control: int, target1: int, target2: int) -> QuantumCircuit:
        """Apply controlled-SWAP (Fredkin) gate."""
        self.append(CSwapGate(), [control, target1, target2])
        return self

    def fredkin(self, control: int, target1: int, target2: int) -> QuantumCircuit:
        """Alias for :meth:`cswap`."""
        return self.cswap(control, target1, target2)

    def ccz(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Apply CCZ gate."""
        self.append(CCZGate(), [control1, control2, target])
        return self

    def mcx(self, controls: Sequence[int], target: int) -> QuantumCircuit:
        """Apply multi-controlled X gate."""
        nc = len(controls)
        self.append(MCXGate(num_controls=nc), list(controls) + [target])
        return self

    def mcz(self, controls: Sequence[int], target: int) -> QuantumCircuit:
        """Apply multi-controlled Z gate."""
        nc = len(controls)
        self.append(MCZGate(num_controls=nc), list(controls) + [target])
        return self

    def dcx(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply DCX (double-CNOT) gate."""
        self.append(DCXGate(), [qubit1, qubit2])
        return self

    def ms(self, theta: float, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply Mølmer–Sørensen gate."""
        self.append(MSGate(params=[theta]), [qubit1, qubit2], [theta])
        return self

    # -- Measurement ----------------------------------------------------------

    def measure(
        self,
        qubit: Union[int, Sequence[int]],
        cbit: Union[int, Sequence[int]],
    ) -> QuantumCircuit:
        """Measure qubit(s) into classical bit(s).

        Parameters
        ----------
        qubit : int or sequence of int
            Qubit indices to measure.
        cbit : int or sequence of int
            Classical bit indices to store results.

        Returns
        -------
        QuantumCircuit
            self
        """
        if isinstance(qubit, int):
            qubits = [qubit]
            cbits = [cbit]
        else:
            qubits = list(qubit)
            cbits = list(cbit)

        if len(qubits) != len(cbits):
            raise ValueError(
                f"Number of qubits ({len(qubits)}) must equal "
                f"number of classical bits ({len(cbits)})"
            )

        for q, c in zip(qubits, cbits):
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self.num_qubits})")
            if c < 0 or c >= self.num_clbits:
                raise ValueError(f"Classical bit index {c} out of range [0, {self.num_clbits})")

        for q in qubits:
            self.append(Measurement(num_qubits=1), [q])

        return self

    # -- Barrier & Reset ------------------------------------------------------

    def barrier(self, *qubits: int) -> QuantumCircuit:
        """Insert a barrier.

        Parameters
        ----------
        *qubits : int
            Qubits covered by the barrier. If none given, barrier spans all qubits.
        """
        if qubits:
            self._data.append(Barrier(qubits=list(qubits)))
        else:
            self._data.append(Barrier(qubits=list(range(self.num_qubits))))
        return self

    def reset(self, qubit: int) -> QuantumCircuit:
        """Reset a qubit to |0⟩."""
        self._data.append(Reset(qubits=qubit))
        return self

    # -- General append -------------------------------------------------------

    def append(
        self,
        gate: Union[Gate, Measurement],
        qubits: Optional[Sequence[int]] = None,
        params: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
    ) -> QuantumCircuit:
        """Append a gate (or measurement) to the circuit.

        Parameters
        ----------
        gate : Gate or Measurement
        qubits : sequence of int, optional
            Target qubit indices.
        params : sequence of float, optional
            Parameter values for parameterised gates.
        label : str, optional
            Custom display label.

        Returns
        -------
        QuantumCircuit
            self
        """
        if isinstance(gate, Measurement):
            op = Operation(gate, qubits=qubits, label=label)
            self._data.append(op)
            return self

        if not isinstance(gate, Gate):
            raise TypeError(f"Expected Gate or Measurement, got {type(gate)}")

        # Validate qubits
        qubit_list = list(qubits) if qubits is not None else []
        for q in qubit_list:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {self.num_qubits})"
                )

        # Validate params
        param_list = list(params) if params is not None else []
        if len(param_list) != gate.num_params:
            raise ValueError(
                f"Gate '{gate.name}' expects {gate.num_params} params, "
                f"got {len(param_list)}"
            )

        op = Operation(
            gate=gate,
            qubits=qubit_list,
            params=param_list,
            label=label,
        )
        self._data.append(op)
        return self

    # -- Circuit composition --------------------------------------------------

    def compose(self, other: QuantumCircuit, inplace: bool = False) -> QuantumCircuit:
        """Compose with another circuit (sequential concatenation).

        ``self.compose(other)`` produces ``other ∘ self`` (other runs *after* self).

        Parameters
        ----------
        other : QuantumCircuit
        inplace : bool
            If ``True``, modify this circuit. Otherwise return a new one.

        Returns
        -------
        QuantumCircuit
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(
                f"Circuit widths must match: {self.num_qubits} vs {other.num_qubits}"
            )

        target = self if inplace else self.copy()

        # Map other's qubit indices to this circuit's qubits
        for op in other._data:
            new_op = Operation(
                gate=op.gate,
                qubits=op.qubits,
                params=op.params,
                condition=op.condition,
                label=op.label,
            )
            target._data.append(new_op)

        if not inplace:
            return target
        return self

    def tensor(self, other: QuantumCircuit) -> QuantumCircuit:
        """Tensor (parallel) composition with another circuit.

        The resulting circuit acts on ``self.n + other.n`` qubits,
        with self occupying the lower indices.

        Parameters
        ----------
        other : QuantumCircuit

        Returns
        -------
        QuantumCircuit
        """
        result = QuantumCircuit(self.num_qubits + other.num_qubits)

        # Append self's operations (qubits unchanged)
        for op in self._data:
            result._data.append(Operation(
                gate=op.gate,
                qubits=op.qubits,
                params=op.params,
                label=op.label,
            ))

        # Append other's operations (offset qubits by self.num_qubits)
        offset = self.num_qubits
        for op in other._data:
            shifted_qubits = tuple(q + offset for q in op.qubits)
            result._data.append(Operation(
                gate=op.gate,
                qubits=shifted_qubits,
                params=op.params,
                label=op.label,
            ))

        return result

    def parallel(self, other: QuantumCircuit) -> QuantumCircuit:
        """Alias for :meth:`tensor`."""
        return self.tensor(other)

    # -- Inverse --------------------------------------------------------------

    def inverse(self) -> QuantumCircuit:
        """Return the inverse (adjoint) circuit.

        Gates are reversed in order and each gate is replaced by its inverse.

        Returns
        -------
        QuantumCircuit
        """
        result = QuantumCircuit(*self._qregs, *self._cregs, name=f"{self._name}_dg")
        for op in reversed(self._data):
            result._data.append(op.inverse())
        return result

    # -- Unitary matrix -------------------------------------------------------

    def to_unitary(self) -> np.ndarray:
        """Compute the full unitary matrix of the circuit.

        Composes all gate matrices in order via matrix multiplication.
        Only works for circuits without measurements or non-unitary ops.

        Returns
        -------
        numpy.ndarray
            Complex128 matrix of shape ``(2**n, 2**n)``.

        Raises
        ------
        ValueError
            If the circuit contains measurements or resets.
        """
        dim = 1 << self.num_qubits
        result = np.eye(dim, dtype=np.complex128)

        for op in self._data:
            if isinstance(op.gate, Measurement):
                raise ValueError("Cannot compute unitary: circuit contains measurements")
            if isinstance(op, (Barrier, Reset)):
                continue

            gate_matrix = op.gate.to_matrix(*op.params)
            # Place gate into the full Hilbert space
            full_matrix = self._embed_gate(gate_matrix, op.qubits)
            result = full_matrix @ result

        return result

    def to_matrix(self) -> np.ndarray:
        """Alias for :meth:`to_unitary`."""
        return self.to_unitary()

    def _embed_gate(
        self,
        gate_matrix: np.ndarray,
        qubits: Tuple[int, ...],
    ) -> np.ndarray:
        """Embed a gate matrix into the full Hilbert space.

        The gate acts on ``qubits`` within the ``n``-qubit system.
        Uses tensor-product construction with identity on idle qubits.

        Parameters
        ----------
        gate_matrix : numpy.ndarray
            Gate matrix of shape ``(2**k, 2**k)``.
        qubits : tuple of int
            Target qubit indices.

        Returns
        -------
        numpy.ndarray
            Full matrix of shape ``(2**n, 2**n)``.
        """
        n = self.num_qubits
        k = len(qubits)
        dim = 1 << n

        if k == n:
            return gate_matrix

        if k == 0:
            return np.eye(dim, dtype=np.complex128)

        # Build full matrix via iterative Kronecker products
        # Sort qubits and figure out the order
        sorted_indices = sorted(range(n), key=lambda i: (
            qubits.index(i) if i in qubits else float('inf')
        ))

        # Build as a product of single-qubit operators
        full = np.eye(1, dtype=np.complex128)
        gate_qubits_sorted = sorted(qubits)

        # Map from position in sorted qubit list to position in gate matrix
        for q in range(n):
            if q in qubits:
                # Find which position this qubit has in the gate
                pos = qubits.index(q)
                gate_dim = 1 << k
                # Extract single-qubit component from gate
                # This is complex for multi-qubit gates; use the embedding approach
                pass
            else:
                full = np.kron(full, np.eye(2, dtype=np.complex128))

        # Alternative: use the permutation approach
        # Sort the qubits the gate acts on and permute the gate matrix
        # Then tensor with identities in the correct positions

        # Simpler approach: build via repeated Kronecker products
        # Work from right to left
        qubit_list = list(range(n))
        result = None

        # Build factor list: for each qubit from 0 to n-1, either I or part of gate
        # For a multi-qubit gate, this requires splitting the gate matrix
        # Use the swap-based approach instead

        # Most efficient: use the general method
        # Expand gate from acting on `qubits` to acting on all `n` qubits
        full_matrix = np.eye(dim, dtype=np.complex128)

        # We'll use the reshape-and-multiply approach
        # For each qubit the gate acts on, we apply the correct operations

        # Actually, the simplest correct approach:
        # Use the identity: U_full = S^† (U ⊗ I) S where S is a permutation
        # that brings the target qubits to the front

        # Step 1: Determine permutation to bring target qubits to positions 0..k-1
        perm = list(qubits) + [q for q in range(n) if q not in qubits]
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i

        # Step 2: Build permuted space: U ⊗ I_(n-k)
        gate_on_front = np.kron(gate_matrix, np.eye(1 << (n - k), dtype=np.complex128))

        # Step 3: Apply permutation matrices
        perm_matrix = self._permutation_matrix(perm, n)
        inv_perm_matrix = self._permutation_matrix(inv_perm, n)

        return inv_perm_matrix @ gate_on_front @ perm_matrix

    def _permutation_matrix(self, perm: List[int], n: int) -> np.ndarray:
        """Build the permutation matrix for qubit reordering.

        Parameters
        ----------
        perm : list of int
            perm[i] = which original qubit goes to position i.
        n : int
            Number of qubits.

        Returns
        -------
        numpy.ndarray
            Unitary permutation matrix.
        """
        dim = 1 << n
        P = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            # i in binary has bits for the permuted ordering
            # Convert to original ordering
            original = 0
            for pos in range(n):
                original_qubit = perm[pos]
                bit = (i >> (n - 1 - pos)) & 1
                original |= (bit << (n - 1 - original_qubit))
            P[original, i] = 1.0
        return P

    # -- Circuit statistics ---------------------------------------------------

    def depth(self) -> int:
        """Compute circuit depth (longest path of non-parallel gates).

        Returns
        -------
        int
        """
        if not self._data:
            return 0

        # Track which qubits each gate layer uses
        occupied: Set[int] = set()
        depth = 0

        for op in self._data:
            if isinstance(op, (Barrier, Reset)):
                # Barriers and resets count as a layer
                if op.qubits:
                    if any(q in occupied for q in op.qubits):
                        occupied.clear()
                        depth += 1
                    occupied.update(op.qubits)
                continue

            op_qubits = set(op.qubits)
            if op_qubits & occupied:
                occupied.clear()
                depth += 1
            occupied.update(op_qubits)

        if occupied:
            depth += 1

        return depth

    def size(self) -> int:
        """Total number of gate operations (excluding barriers and resets)."""
        return sum(
            1 for op in self._data
            if not isinstance(op, (Barrier, Reset))
        )

    def count_gates(self) -> Dict[str, int]:
        """Count occurrences of each gate type.

        Returns
        -------
        dict
            Mapping from gate name to count.
        """
        counts: Dict[str, int] = {}
        for op in self._data:
            if isinstance(op, (Barrier, Reset)):
                name = op.name
            else:
                name = op.gate.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    # -- Copy & Reverse -------------------------------------------------------

    def copy(self, name: Optional[str] = None) -> QuantumCircuit:
        """Return a deep copy of the circuit.

        Parameters
        ----------
        name : str, optional
            Override the name.

        Returns
        -------
        QuantumCircuit
        """
        result = QuantumCircuit(*self._qregs, *self._cregs, name=name or self._name)
        result._data = [copy.deepcopy(op) for op in self._data]
        result._metadata = copy.deepcopy(self._metadata)
        result._parameters = dict(self._parameters)
        return result

    def reverse(self) -> QuantumCircuit:
        """Return a copy with operations in reverse order (no inverse)."""
        result = self.copy(name=f"{self._name}_rev")
        result._data = list(reversed(result._data))
        return result

    # -- Parameter binding ----------------------------------------------------

    def bind_parameters(self, params: Dict[Union[str, Parameter], float]) -> QuantumCircuit:
        """Bind parameter values to produce a concrete circuit.

        Parameters
        ----------
        params : dict
            Mapping from parameter name (or :class:`Parameter`) to value.

        Returns
        -------
        QuantumCircuit
            New circuit with all parameters resolved.
        """
        param_map: Dict[str, float] = {}
        for k, v in params.items():
            key = k.name if isinstance(k, Parameter) else str(k)
            param_map[key] = float(v)

        result = self.copy()
        new_data: List[Operation] = []
        for op in result._data:
            if isinstance(op, (Barrier, Reset)):
                new_data.append(op)
                continue
            if op.params:
                new_params = list(op.params)
                for i, p in enumerate(new_params):
                    if isinstance(p, Parameter):
                        if p.name in param_map:
                            new_params[i] = param_map[p.name]
                        else:
                            raise ValueError(f"Unbound parameter: {p.name}")
                new_op = Operation(
                    gate=op.gate,
                    qubits=op.qubits,
                    params=new_params,
                    condition=op.condition,
                    label=op.label,
                )
                new_data.append(new_op)
            else:
                new_data.append(op)
        result._data = new_data
        return result

    # -- QASM export ----------------------------------------------------------

    def qasm(self) -> str:
        """Export the circuit to an OpenQASM 2.0 string.

        Returns
        -------
        str
            Valid QASM 2.0 code.

        Raises
        ------
        ValueError
            If the circuit contains classical conditions or unsupported ops.
        """
        lines: List[str] = ["OPENQASM 2.0;", 'include "qelib1.inc";', ""]

        # Register declarations
        for qr in self._qregs:
            lines.append(qr.qasm())
        for cr in self._cregs:
            lines.append(cr.qasm())
        lines.append("")

        # Operations
        for op in self._data:
            if isinstance(op, Barrier):
                lines.append(f"barrier {','.join(f'q[{q}]' for q in op.qubits)};")
            elif isinstance(op, Reset):
                lines.append(f"reset q[{op.qubits[0]}];")
            elif isinstance(op.gate, Measurement):
                q = op.qubits[0]
                # Assume cbit = q (convention for single measurements)
                lines.append(f"measure q[{q}] -> c[{q}];")
            else:
                qasm_str = self._gate_to_qasm(op)
                lines.append(qasm_str)

        return "\n".join(lines)

    def _gate_to_qasm(self, op: Operation) -> str:
        """Convert a single gate operation to QASM."""
        gate = op.gate
        name = op.name if op.name else gate.name
        qargs = ",".join(f"q[{q}]" for q in op.qubits)

        # Map internal names to QASM names
        qasm_name_map = {
            "cx": "cx",
            "h": "h",
            "x": "x",
            "y": "y",
            "z": "z",
            "s": "s",
            "sdg": "sdg",
            "t": "t",
            "tdg": "tdg",
            "rx": "rx",
            "ry": "ry",
            "rz": "rz",
            "u1": "u1",
            "u2": "u2",
            "u3": "u3",
            "u": "u3",
            "cz": "cz",
            "swap": "swap",
            "ccx": "ccx",
            "cswap": "cswap",
            "p": "u1",
            "crx": "crx",
            "cry": "cry",
            "crz": "crz",
        }

        qasm_name = qasm_name_map.get(name, name)

        if op.params:
            params_str = ",".join(f"{p:.15g}" for p in op.params)
            return f"{qasm_name}({params_str}) {qargs};"
        return f"{qasm_name} {qargs};"

    # -- Factory methods ------------------------------------------------------

    @classmethod
    def from_gates(
        cls,
        gates: Sequence[Tuple[Gate, Sequence[int], Sequence[float]]],
        num_qubits: int,
        num_clbits: int = 0,
        name: Optional[str] = None,
    ) -> QuantumCircuit:
        """Create a circuit from a list of (gate, qubits, params) tuples.

        Parameters
        ----------
        gates : sequence of (Gate, qubits, params)
        num_qubits : int
        num_clbits : int, optional
        name : str, optional

        Returns
        -------
        QuantumCircuit
        """
        qc = cls(num_qubits, name=name)
        if num_clbits > 0:
            qc.add_register(ClassicalRegister(num_clbits))
        for gate, qubits, params in gates:
            qc.append(gate, qubits, params)
        return qc

    @classmethod
    def random(
        cls,
        num_qubits: int,
        depth: int,
        seed: Optional[int] = None,
        max_operands: int = 3,
        gate_pool: Optional[Sequence[Gate]] = None,
        name: Optional[str] = None,
    ) -> QuantumCircuit:
        """Generate a random circuit.

        Parameters
        ----------
        num_qubits : int
            Number of qubits.
        depth : int
            Number of gate layers.
        seed : int, optional
            Random seed.
        max_operands : int, optional
            Maximum number of qubits per gate. Default 3.
        gate_pool : sequence of Gate, optional
            Gates to choose from. Default includes H, X, Y, Z, RX, RY,
            RZ, CX, CZ, SWAP.
        name : str, optional

        Returns
        -------
        QuantumCircuit
        """
        rng = random.Random(seed)

        if gate_pool is None:
            gate_pool = [
                HGate(), XGate(), YGate(), ZGate(),
                RXGate(params=[0.0]), RYGate(params=[0.0]), RZGate(params=[0.0]),
                CNOTGate(), CZGate(), SwapGate(),
            ]

        qc = cls(num_qubits, name=name or f"random_{depth}")

        for _ in range(depth):
            gate = rng.choice(gate_pool)
            n_ops = min(gate.num_qubits, max_operands)

            if n_ops == 1:
                qubit = rng.randint(0, num_qubits - 1)
                if gate.num_params > 0:
                    angle = rng.uniform(0, 2 * math.pi)
                    param_gate = type(gate)(params=[angle] * gate.num_params)
                    qc.append(param_gate, [qubit], [angle] * gate.num_params)
                else:
                    qc.append(gate, [qubit])
            elif n_ops == 2:
                qubits = rng.sample(range(num_qubits), 2)
                if gate.num_params > 0:
                    angle = rng.uniform(0, 2 * math.pi)
                    param_gate = type(gate)(params=[angle] * gate.num_params)
                    qc.append(param_gate, qubits, [angle] * gate.num_params)
                else:
                    qc.append(gate, qubits)
            elif n_ops == 3:
                qubits = rng.sample(range(num_qubits), 3)
                qc.append(gate, qubits)

        return qc

    # -- String representation ------------------------------------------------

    def __repr__(self) -> str:
        regs = ""
        if self._qregs:
            regs += f" qregs={self._qregs}"
        if self._cregs:
            regs += f" cregs={self._cregs}"
        return (
            f"QuantumCircuit(name={self._name!r}, "
            f"num_qubits={self.num_qubits}, "
            f"num_clbits={self.num_clbits}, "
            f"num_ops={len(self._data)}{regs})"
        )

    def __str__(self) -> str:
        """ASCII art circuit diagram."""
        if not self._data:
            return f"<QuantumCircuit: {self.num_qubits} qubits, 0 operations>"

        n = self.num_qubits
        if n == 0:
            return "<QuantumCircuit: 0 qubits>"

        # Each qubit gets a row
        rows: List[List[str]] = [[] for _ in range(n)]

        for op in self._data:
            if isinstance(op, Barrier):
                for q in op.qubits:
                    rows[q].append("───")
                # Add vertical bars
                min_q = min(op.qubits)
                max_q = max(op.qubits)
                if min_q != max_q:
                    rows[min_q].append("║")
                    rows[max_q].append("║")
                continue

            if isinstance(op, Reset):
                q = op.qubits[0]
                rows[q].append("─|0>─")
                for other_q in range(n):
                    if other_q != q:
                        rows[other_q].append("─────")
                continue

            if isinstance(op.gate, Measurement):
                q = op.qubits[0]
                rows[q].append(f"─M─→c[{q}]")
                for other_q in range(n):
                    if other_q != q:
                        rows[other_q].append("─────────")
                continue

            gate_label = op.gate.name
            if op.params:
                param_strs = [f"{p:.3g}" for p in op.params]
                gate_label += f"({','.join(param_strs)})"

            qubits = op.qubits
            if len(qubits) == 1:
                q = qubits[0]
                label = gate_label.center(7)
                rows[q].append(f"─{label}─")
                for other_q in range(n):
                    if other_q != q:
                        rows[other_q].append("───────")
            elif len(qubits) == 2:
                ctrl, tgt = qubits
                for other_q in range(n):
                    if other_q == ctrl and other_q == tgt:
                        rows[other_q].append(f"──{gate_label}──")
                    elif other_q == ctrl:
                        rows[other_q].append(f"──■─────")
                    elif other_q == tgt:
                        rows[other_q].append(f"──{gate_label}──")
                    else:
                        rows[other_q].append("────────")
            elif len(qubits) >= 3:
                for other_q in range(n):
                    if other_q == qubits[-1]:
                        rows[other_q].append(f"──{gate_label}──")
                    elif other_q in qubits[:-1]:
                        rows[other_q].append("──■─────")
                    else:
                        rows[other_q].append("────────")

        # Build final string
        lines = []
        for i in range(n):
            q_label = f"q[{i}]: "
            row_str = "".join(rows[i]) if rows[i] else "────"
            lines.append(f"{q_label}{row_str}")

        header = f"QuantumCircuit '{self._name}' — {self.num_qubits} qubits, {len(self._data)} ops"
        lines.insert(0, header)
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumCircuit):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            return False
        if len(self._data) != len(other._data):
            return False
        return all(a == b for a, b in zip(self._data, other._data))

    # -- Iteration ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index: int) -> Operation:
        return self._data[index]
