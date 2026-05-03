"""
Quantum Operations
==================

Defines the operation layer that sits between gates and circuits.  An
:class:`Operation` bundles a gate (or special instruction) together with the
qubits it acts on, its parameters, and an optional classical condition.

This module provides:

* :class:`Operation` — atomic gate application with qubits and parameters.
* :class:`CompositeOperation` — a sequence of operations (macro / sub-circuit).
* :class:`Barrier` — a visual / scheduling barrier between gate layers.
* :class:`Reset` — reset a qubit to |0⟩.
* :class:`ConditionalOperation` — execute an operation conditioned on classical bits.

Typical usage::

    >>> from quantumflow.core.gate import XGate, HGate
    >>> op = Operation(HGate(), qubits=[0])
    >>> print(op)
    HGate on qubits [0]

    >>> comp = CompositeOperation([
    ...     Operation(HGate(), qubits=[0]),
    ...     Operation(XGate(), qubits=[0]),
    ... ])
    >>> print(comp)
    CompositeOperation(2 operations)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    from quantumflow.core.gate import Gate

__all__ = [
    "Operation",
    "CompositeOperation",
    "Barrier",
    "Reset",
    "ConditionalOperation",
    "Instruction",
]


# ---------------------------------------------------------------------------
# Abstract instruction base
# ---------------------------------------------------------------------------

class Instruction(ABC):
    """Abstract base class for any instruction in a quantum circuit.

    Instructions are the most generic building block — they include gates,
    barriers, resets, measurements, and composite blocks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this instruction."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits this instruction touches."""

    def inverse(self) -> Instruction:
        """Return the inverse of this instruction.

        The default implementation returns ``self`` (idempotent for
        barriers, resets, etc.).
        """
        return self

    def copy(self) -> Instruction:
        """Return a deep copy of this instruction."""
        return copy.deepcopy(self)

    @abstractmethod
    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Operation
# ---------------------------------------------------------------------------

class Operation(Instruction):
    """A gate applied to specific qubits with specific parameters.

    This is the core data structure that a :class:`~quantumflow.core.circuit.QuantumCircuit`
    stores internally for every gate application.

    Parameters
    ----------
    gate : Gate
        The quantum gate to apply.
    qubits : list of int
        Indices of the target qubits.
    params : list of float, optional
        Numerical values for any parameterised gates.
    condition : tuple of (str, int), optional
        Classical condition ``(register_name, value)`` that gates on
        the register equaling ``value``.
    label : str, optional
        Custom display label, overriding the gate's name.

    Attributes
    ----------
    gate : Gate
    qubits : tuple of int
    params : tuple of float
    condition : tuple or None
    label : str or None

    Examples
    --------
    >>> from quantumflow.core.gate import HGate, CNOTGate
    >>> op = Operation(HGate(), qubits=[0])
    >>> op.num_qubits
    1
    >>> op2 = Operation(CNOTGate(), qubits=[0, 1])
    >>> op2.num_qubits
    2
    """

    __slots__ = (
        "_gate",
        "_qubits",
        "_params",
        "_condition",
        "_label",
    )

    def __init__(
        self,
        gate: Gate,
        qubits: Optional[Sequence[int]] = None,
        params: Optional[Sequence[float]] = None,
        condition: Optional[Tuple[str, int]] = None,
        label: Optional[str] = None,
    ) -> None:
        self._gate = gate
        self._qubits: Tuple[int, ...] = tuple(qubits) if qubits is not None else ()
        self._params: Tuple[float, ...] = tuple(params) if params is not None else ()
        self._condition: Optional[Tuple[str, int]] = condition
        self._label: Optional[str] = label

        # Validate
        if len(self._qubits) != 0 and len(self._qubits) != self._gate.num_qubits:
            raise ValueError(
                f"Gate '{self._gate.name}' acts on {self._gate.num_qubits} qubits, "
                f"but {len(self._qubits)} qubits were provided: {self._qubits}"
            )

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        """str: Display name (uses label if set, otherwise gate name)."""
        if self._label is not None:
            return self._label
        return self._gate.name

    @property
    def gate(self) -> Gate:
        """Gate: The underlying gate."""
        return self._gate

    @property
    def qubits(self) -> Tuple[int, ...]:
        """tuple of int: Target qubit indices."""
        return self._qubits

    @property
    def params(self) -> Tuple[float, ...]:
        """tuple of float: Bound parameter values."""
        return self._params

    @property
    def num_qubits(self) -> int:
        """int: Number of target qubits."""
        return len(self._qubits)

    @property
    def condition(self) -> Optional[Tuple[str, int]]:
        """tuple or None: Classical condition ``(register_name, value)``."""
        return self._condition

    @property
    def is_conditional(self) -> bool:
        """bool: Whether this operation is classically conditioned."""
        return self._condition is not None

    @property
    def label(self) -> Optional[str]:
        """str or None: Custom display label."""
        return self._label

    @label.setter
    def label(self, value: Optional[str]) -> None:
        self._label = value

    # -- Inverse -------------------------------------------------------------

    def inverse(self) -> Operation:
        """Return an operation that undoes this one.

        Returns
        -------
        Operation
            New operation with the inverse gate on the same qubits.
        """
        inv_gate = self._gate.inverse()
        return Operation(
            gate=inv_gate,
            qubits=self._qubits,
            params=self._params,
            condition=self._condition,
            label=f"{self._label + '†' if self._label else self.name}†" if self._label is not None else None,
        )

    # -- Matrix ---------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Return the unitary matrix of this operation.

        Returns
        -------
        numpy.ndarray
            Complex128 matrix of shape ``(2**n, 2**n)``.
        """
        return self._gate.to_matrix(*self._params)

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{self._gate.__class__.__name__}"]
        if self._qubits:
            parts.append(f"on qubits {list(self._qubits)}")
        if self._params:
            parts.append(f"params={list(self._params)}")
        if self._condition:
            parts.append(f"if {self._condition[0]}=={self._condition[1]}")
        return "Operation(" + ", ".join(parts) + ")"

    def __str__(self) -> str:
        cond_str = ""
        if self._condition:
            cond_str = f" [if {self._condition[0]}=={self._condition[1]}]"
        return f"{self.name} on qubits {list(self._qubits)}{cond_str}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operation):
            return NotImplemented
        return (
            self._gate == other._gate
            and self._qubits == other._qubits
            and self._params == other._params
            and self._condition == other._condition
        )

    def __hash__(self) -> int:
        return hash((self._gate, self._qubits, self._params, self._condition))


# ---------------------------------------------------------------------------
# CompositeOperation
# ---------------------------------------------------------------------------

class CompositeOperation(Instruction):
    """A sequence of operations grouped as a single block.

    Composite operations act as a macro: they can be appended to circuits,
    inverted, and copied as a unit.

    Parameters
    ----------
    operations : list of Operation or Instruction
        The operations to group together.
    label : str, optional
        Human-readable label.

    Examples
    --------
    >>> from quantumflow.core.gate import HGate, XGate
    >>> comp = CompositeOperation([
    ...     Operation(HGate(), [0]),
    ...     Operation(XGate(), [0]),
    ... ])
    >>> len(comp)
    2
    """

    def __init__(
        self,
        operations: Optional[Sequence[Operation]] = None,
        label: Optional[str] = None,
    ) -> None:
        self._operations: List[Operation] = list(operations) if operations else []
        self._label: Optional[str] = label

    @property
    def name(self) -> str:
        """str: Display name."""
        if self._label:
            return self._label
        return "CompositeOperation"

    @property
    def num_qubits(self) -> int:
        """int: Maximum number of qubits used by any sub-operation."""
        if not self._operations:
            return 0
        return max(op.num_qubits for op in self._operations)

    @property
    def operations(self) -> List[Operation]:
        """list of Operation: The constituent operations."""
        return list(self._operations)

    # -- Mutating methods -----------------------------------------------------

    def append(self, operation: Operation) -> None:
        """Add an operation to the end of this composite.

        Parameters
        ----------
        operation : Operation
        """
        self._operations.append(operation)

    def extend(self, operations: Sequence[Operation]) -> None:
        """Add multiple operations.

        Parameters
        ----------
        operations : sequence of Operation
        """
        self._operations.extend(operations)

    def insert(self, index: int, operation: Operation) -> None:
        """Insert an operation at position ``index``."""
        self._operations.insert(index, operation)

    def remove(self, operation: Operation) -> None:
        """Remove the first occurrence of ``operation``."""
        self._operations.remove(operation)

    def clear(self) -> None:
        """Remove all operations."""
        self._operations.clear()

    # -- Inverse -------------------------------------------------------------

    def inverse(self) -> CompositeOperation:
        """Return the inverse composite (operations in reverse order, each inverted).

        Returns
        -------
        CompositeOperation
        """
        inv_ops = [op.inverse() for op in reversed(self._operations)]
        return CompositeOperation(inv_ops, label=f"{self.name}†")

    # -- Iteration ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._operations)

    def __iter__(self) -> Iterator[Operation]:
        return iter(self._operations)

    def __getitem__(self, index: int) -> Operation:
        return self._operations[index]

    def __contains__(self, item: object) -> bool:
        return item in self._operations

    def __repr__(self) -> str:
        return (
            f"CompositeOperation({len(self._operations)} operations"
            + (f", label={self._label!r})" if self._label else ")")
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeOperation):
            return NotImplemented
        return self._operations == other._operations

    def flatten(self) -> List[Operation]:
        """Recursively flatten nested composite operations.

        Returns
        -------
        list of Operation
            Flat list of all leaf operations.
        """
        result: List[Operation] = []
        for op in self._operations:
            if isinstance(op, CompositeOperation):
                result.extend(op.flatten())
            else:
                result.append(op)
        return result


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------

class Barrier(Instruction):
    """A circuit barrier that prevents gate reordering across it.

    Barriers have no physical effect — they are purely organisational
    directives for compilers and visualisers.

    Parameters
    ----------
    qubits : sequence of int, optional
        Qubits covered by the barrier. If ``None``, the barrier spans
        all circuit qubits (resolved at circuit level).

    Examples
    --------
    >>> b = Barrier(qubits=[0, 1, 2])
    >>> b.name
    'barrier'
    """

    def __init__(self, qubits: Optional[Sequence[int]] = None) -> None:
        self._qubits: Tuple[int, ...] = tuple(qubits) if qubits is not None else ()

    @property
    def name(self) -> str:
        """str: Always ``'barrier'``."""
        return "barrier"

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits in the barrier."""
        return len(self._qubits)

    @property
    def qubits(self) -> Tuple[int, ...]:
        """tuple of int: Qubits covered by the barrier."""
        return self._qubits

    def __repr__(self) -> str:
        if self._qubits:
            return f"Barrier(qubits={list(self._qubits)})"
        return "Barrier()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Barrier):
            return NotImplemented
        return self._qubits == other._qubits

    def __hash__(self) -> int:
        return hash(("barrier", self._qubits))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class Reset(Instruction):
    """Reset a qubit to the |0⟩ state.

    This is a non-unitary operation that collapses the target qubit.

    Parameters
    ----------
    qubits : sequence of int
        Qubit indices to reset.

    Examples
    --------
    >>> r = Reset(qubits=[2])
    >>> r.name
    'reset'
    """

    def __init__(self, qubits: Union[int, Sequence[int]]) -> None:
        if isinstance(qubits, int):
            qubits = [qubits]
        self._qubits: Tuple[int, ...] = tuple(qubits)

    @property
    def name(self) -> str:
        """str: Always ``'reset'``."""
        return "reset"

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits to reset."""
        return len(self._qubits)

    @property
    def qubits(self) -> Tuple[int, ...]:
        """tuple of int: Qubits to reset."""
        return self._qubits

    def __repr__(self) -> str:
        return f"Reset(qubits={list(self._qubits)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reset):
            return NotImplemented
        return self._qubits == other._qubits

    def __hash__(self) -> int:
        return hash(("reset", self._qubits))


# ---------------------------------------------------------------------------
# ConditionalOperation
# ---------------------------------------------------------------------------

class ConditionalOperation(Instruction):
    """Wrap an operation with a classical condition.

    The operation is only executed when the specified classical register
    (or bit) equals a given integer value.

    Parameters
    ----------
    operation : Operation or Instruction
        The operation to guard.
    register_name : str
        Name of the classical register to check.
    value : int
        The register must equal this value for the operation to execute.

    Examples
    --------
    >>> from quantumflow.core.gate import XGate
    >>> cond = ConditionalOperation(
    ...     Operation(XGate(), [0]),
    ...     register_name='c',
    ...     value=1,
    ... )
    >>> cond
    ConditionalOperation(XGate, if c==1)
    """

    def __init__(
        self,
        operation: Instruction,
        register_name: str,
        value: int,
    ) -> None:
        self._operation = operation
        self._register_name = register_name
        self._value = value

    @property
    def name(self) -> str:
        """str: Display name including condition."""
        return f"{self._operation.name} (if {self._register_name}=={self._value})"

    @property
    def num_qubits(self) -> int:
        """int: Number of qubits in the guarded operation."""
        return self._operation.num_qubits

    @property
    def operation(self) -> Instruction:
        """Instruction: The guarded operation."""
        return self._operation

    @property
    def condition(self) -> Tuple[str, int]:
        """tuple of (str, int): ``(register_name, value)``."""
        return (self._register_name, self._value)

    @property
    def register_name(self) -> str:
        """str: Classical register name."""
        return self._register_name

    @property
    def value(self) -> int:
        """int: Required register value."""
        return self._value

    def inverse(self) -> ConditionalOperation:
        """Return a conditional operation wrapping the inverse."""
        return ConditionalOperation(
            self._operation.inverse(),
            self._register_name,
            self._value,
        )

    def __repr__(self) -> str:
        return (
            f"ConditionalOperation({self._operation.__class__.__name__}, "
            f"if {self._register_name}=={self._value})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConditionalOperation):
            return NotImplemented
        return (
            self._operation == other._operation
            and self._register_name == other._register_name
            and self._value == other._value
        )

    def __hash__(self) -> int:
        return hash(("conditional", self._operation, self._register_name, self._value))
