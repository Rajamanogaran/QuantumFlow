"""
Quantum and Classical Register Management
==========================================

Provides :class:`QuantumRegister` and :class:`ClassicalRegister` for managing
collections of qubits and classical bits within a quantum circuit.

Each register supports indexing, slicing, iteration, and concatenation,
mirroring the interface used in production frameworks like Qiskit.

Typical usage::

    >>> qr = QuantumRegister(4, name='q')
    >>> cr = ClassicalRegister(4, name='c')
    >>> qc = QuantumCircuit(qr, cr)
    >>> qr[0]          # first qubit in the register
    >>> qr[1:3]        # slice of qubits
    >>> list(qr)       # iterate over all qubits
    >>> len(qr)         # number of qubits (4)

See Also
--------
    quantumflow.core.qubit.Qubit : individual qubit representation
    quantumflow.core.circuit.QuantumCircuit : circuit container
"""

from __future__ import annotations

import re
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "RegisterSizeError",
    "QuantumRegister",
    "ClassicalRegister",
    "Register",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RegisterSizeError(ValueError):
    """Raised when a register operation violates size constraints.

    Examples
    --------
    >>> raise RegisterSizeError("Cannot create register with negative size")
    RegisterSizeError: Cannot create register with negative size
    """

    def __init__(self, message: str = "Invalid register size") -> None:
        self.message = message
        super().__init__(self.message)


class RegisterIndexError(IndexError):
    """Raised when an index is out of bounds for a register.

    Parameters
    ----------
    index : int
        The offending index.
    size : int
        The size of the register.
    register_name : str, optional
        Name of the register for context.
    """

    def __init__(
        self,
        index: int,
        size: int,
        register_name: str = "",
    ) -> None:
        self.index = index
        self.size = size
        self.register_name = register_name
        name_part = f" '{register_name}'" if register_name else ""
        super().__init__(
            f"Index {index} out of range for register{name_part} of size {size}."
        )


class RegisterConflictError(ValueError):
    """Raised when two registers have conflicting names or qubit indices."""

    def __init__(self, message: str = "Register conflict") -> None:
        self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Register base mixin
# ---------------------------------------------------------------------------

class Register(Sequence[int]):
    """Abstract base mixin for quantum and classical registers.

    Provides common functionality for indexing, slicing, iteration,
    and string representation shared by both register types.

    Parameters
    ----------
    size : int
        Number of elements (qubits or classical bits) in the register.
    name : str, optional
        Human-readable identifier for the register. Must be a valid
        Python identifier when used in OpenQASM export.
    """

    _DEFAULT_NAME_Q = "q"
    _DEFAULT_NAME_C = "c"
    _VALID_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def __init__(self, size: int, name: Optional[str] = None) -> None:
        if not isinstance(size, int) or size < 0:
            raise RegisterSizeError(
                f"Register size must be a non-negative integer, got {size!r}"
            )
        self._size: int = size
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Register name must be a string or None, got {type(name)}")
        if name is not None and not self._VALID_NAME_RE.match(name):
            raise ValueError(
                f"Register name '{name}' is not a valid identifier. "
                "Use only letters, digits, and underscores, starting with a letter or underscore."
            )
        self._name: str = name if name is not None else ""

    # -- Properties ----------------------------------------------------------

    @property
    def size(self) -> int:
        """int: Number of elements in the register."""
        return self._size

    @property
    def name(self) -> str:
        """str: Human-readable name of the register."""
        return self._name

    # -- Sequence interface --------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> List[int]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        if isinstance(index, int):
            if index < -self._size or index >= self._size:
                raise RegisterIndexError(index, self._size, self._name)
            return index if index >= 0 else index + self._size
        elif isinstance(index, slice):
            start, stop, step = index.indices(self._size)
            return list(range(start, stop, step))
        raise TypeError(f"Indices must be integers or slices, not {type(index)}")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._size))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, int):
            return 0 <= item < self._size
        return False

    # -- Rich comparison -----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return NotImplemented
        return self._size == other._size and self._name == other._name

    def __hash__(self) -> int:
        return hash((self._size, self._name))

    # -- String representation ------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._size}, name={self._name!r})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._size}, '{self._name}')"


# ---------------------------------------------------------------------------
# QuantumRegister
# ---------------------------------------------------------------------------

class QuantumRegister(Register):
    """A register of :math:`n` qubits.

    A ``QuantumRegister`` holds a contiguous group of qubits that can be
    addressed by index within a :class:`~quantumflow.core.circuit.QuantumCircuit`.

    The register itself is essentially a sized, named container — the actual
    qubit state is managed by the simulator backend.

    Parameters
    ----------
    size : int
        Number of qubits. Must be non-negative.
    name : str, optional
        Identifier used in circuit diagrams and QASM export.
        Defaults to ``'q'``.

    Attributes
    ----------
    size : int
        Number of qubits in the register.
    name : str
        Register name.

    Examples
    --------
    Create a 3-qubit register:

    >>> qr = QuantumRegister(3, name='alpha')
    >>> len(qr)
    3
    >>> qr[0]
    0
    >>> qr[1:3]
    [1, 2]
    >>> list(qr)
    [0, 1, 2]

    Two registers are equal when their size and name match:

    >>> QuantumRegister(2, 'q') == QuantumRegister(2, 'q')
    True
    >>> QuantumRegister(2, 'q') == QuantumRegister(2, 'r')
    False
    """

    def __init__(self, size: int, name: str = "q") -> None:
        super().__init__(size, name=name)

    # -- QASM export ---------------------------------------------------------

    def qasm(self) -> str:
        """Return an OpenQASM 2.0 declaration string.

        Returns
        -------
        str
            A ``qreg`` declaration, e.g. ``"qreg q[3];"``.

        Examples
        --------
        >>> QuantumRegister(4, 'data').qasm()
        'qreg data[4];'
        """
        return f"qreg {self._name}[{self._size}];"

    # -- Concatenation -------------------------------------------------------

    def __add__(self, other: QuantumRegister) -> QuantumRegister:
        """Concatenate two quantum registers into a new one.

        The resulting register has the combined size. Its name is derived
        from the two operands joined by ``'+'`` (truncated if needed).

        Parameters
        ----------
        other : QuantumRegister
            Register to concatenate.

        Returns
        -------
        QuantumRegister
            New register containing all qubits from both operands.
        """
        if not isinstance(other, QuantumRegister):
            return NotImplemented
        new_name = f"{self._name}+{other._name}"
        # Truncate to 20 chars to keep it manageable
        if len(new_name) > 20:
            new_name = new_name[:17] + "..."
        return QuantumRegister(self._size + other._size, name=new_name)

    def __iadd__(self, other: QuantumRegister) -> QuantumRegister:
        return self.__add__(other)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ClassicalRegister
# ---------------------------------------------------------------------------

class ClassicalRegister(Register):
    """A register of :math:`n` classical bits.

    A ``ClassicalRegister`` is used to store measurement results from
    qubits. Each classical bit holds a single binary value (0 or 1).

    Parameters
    ----------
    size : int
        Number of classical bits. Must be non-negative.
    name : str, optional
        Identifier used in circuit diagrams and QASM export.
        Defaults to ``'c'``.

    Attributes
    ----------
    size : int
        Number of classical bits in the register.
    name : str
        Register name.

    Examples
    --------
    Create a 2-bit classical register:

    >>> cr = ClassicalRegister(2, name='meas')
    >>> len(cr)
    2
    >>> cr[0]
    0
    >>> cr.qasm()
    'creg meas[2];'
    """

    def __init__(self, size: int, name: str = "c") -> None:
        super().__init__(size, name=name)

    # -- QASM export ---------------------------------------------------------

    def qasm(self) -> str:
        """Return an OpenQASM 2.0 declaration string.

        Returns
        -------
        str
            A ``creg`` declaration, e.g. ``"creg c[2];"``.

        Examples
        --------
        >>> ClassicalRegister(8, 'out').qasm()
        'creg out[8];'
        """
        return f"creg {self._name}[{self._size}];"

    # -- Concatenation -------------------------------------------------------

    def __add__(self, other: ClassicalRegister) -> ClassicalRegister:
        """Concatenate two classical registers.

        Parameters
        ----------
        other : ClassicalRegister
            Register to concatenate.

        Returns
        -------
        ClassicalRegister
            New register containing all bits from both operands.
        """
        if not isinstance(other, ClassicalRegister):
            return NotImplemented
        new_name = f"{self._name}+{other._name}"
        if len(new_name) > 20:
            new_name = new_name[:17] + "..."
        return ClassicalRegister(self._size + other._size, name=new_name)

    def __iadd__(self, other: ClassicalRegister) -> ClassicalRegister:
        return self.__add__(other)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def registers_overlap(
    reg_a: Register,
    reg_b: Register,
    offset_a: int = 0,
    offset_b: int = 0,
) -> bool:
    """Check whether two registers overlap in global qubit indices.

    Parameters
    ----------
    reg_a, reg_b : Register
        Registers to compare.
    offset_a, offset_b : int, optional
        Global offsets of each register. Default is 0.

    Returns
    -------
    bool
        ``True`` if the registers share any global qubit indices.
    """
    set_a = set(range(offset_a, offset_a + reg_a.size))
    set_b = set(range(offset_b, offset_b + reg_b.size))
    return bool(set_a & set_b)


def total_register_size(registers: Sequence[Register]) -> int:
    """Return the combined size of a sequence of registers.

    Parameters
    ----------
    registers : sequence of Register
        Registers to sum.

    Returns
    -------
    int
        Total number of elements across all registers.
    """
    return sum(r.size for r in registers)


def validate_register_names(
    registers: Sequence[Register],
) -> Dict[str, Register]:
    """Validate that all register names are unique.

    Parameters
    ----------
    registers : sequence of Register
        Registers whose names should be checked.

    Returns
    -------
    dict
        Mapping from name to register.

    Raises
    ------
    RegisterConflictError
        If duplicate names are found.
    """
    name_map: Dict[str, Register] = {}
    for reg in registers:
        if reg.name in name_map:
            raise RegisterConflictError(
                f"Duplicate register name '{reg.name}'. "
                f"All registers must have unique names."
            )
        name_map[reg.name] = reg
    return name_map
