"""
QuantumFlow Core Module
=======================

The ``quantumflow.core`` package provides the foundational abstractions for
quantum circuit construction, gate definitions, state management, and
register handling.

Submodules
----------
    qubit       — Qubit representation, :class:`QubitState` enum, Bloch vector helpers.
    gate        — 50+ gate definitions, :class:`GateLibrary`, ``controlled``, ``power``, ``dagger``.
    circuit     — :class:`QuantumCircuit` — the central user-facing data structure.
    register    — :class:`QuantumRegister`, :class:`ClassicalRegister`.
    state       — :class:`Statevector`, :class:`DensityMatrix`, :class:`Operator`, :class:`Observable`.
    operation   — :class:`Operation`, :class:`CompositeOperation`, :class:`Barrier`, :class:`Reset`.

Quick start::

    from quantumflow.core import QuantumCircuit, Statevector

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    print(qc)

    sv = Statevector.zero(2)
    sv = sv.evolve(qc.to_unitary())
    print(sv.probabilities())
"""

from quantumflow.core.circuit import (
    QuantumCircuit,
)
from quantumflow.core.gate import (
    CCXGate,
    CXGate,
    CCZGate,
    CNOTGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSwapGate,
    CZGate,
    CompositeGate,
    ControlledGate,
    DCXGate,
    ESCGate,
    FredkinGate,
    Gate,
    GateLibrary,
    GlobalPhaseGate,
    HGate,
    ISwapGate,
    MCXGate,
    MCZGate,
    MSGate,
    Measurement,
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
    SXdgGate,
    SXGate,
    SdgGate,
    SwapGate,
    TGate,
    TdgGate,
    ToffoliGate,
    U1Gate,
    U2Gate,
    U3Gate,
    UGate,
    UnitaryGate,
    XGate,
    XYGate,
    YGate,
    ZGate,
    controlled,
    dagger,
    power,
)
from quantumflow.core.operation import (
    Barrier,
    CompositeOperation,
    ConditionalOperation,
    Operation,
    Reset,
)
from quantumflow.core.qubit import (
    BASIS_ONE,
    BASIS_ZERO,
    MultiQubitState,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    Qubit,
    QubitState,
    QubitStateVector,
    SQRT2,
    SQRT2_INV,
    bloch_vector,
    polarization,
    state_fidelity,
)
from quantumflow.core.register import (
    ClassicalRegister,
    QuantumRegister,
    Register,
    RegisterConflictError,
    RegisterIndexError,
    RegisterSizeError,
    registers_overlap,
    total_register_size,
    validate_register_names,
)
from quantumflow.core.state import (
    DensityMatrix,
    Observable,
    Operator,
    QuantumState,
    Statevector,
)

__all__ = [
    # Circuit
    "QuantumCircuit",
    # Gates — base
    "Gate",
    "UnitaryGate",
    "ParameterizedGate",
    "ControlledGate",
    "CompositeGate",
    "Measurement",
    # Gates — single-qubit
    "HGate",
    "XGate",
    "YGate",
    "ZGate",
    "SGate",
    "SdgGate",
    "TGate",
    "TdgGate",
    "SXGate",
    "SXdgGate",
    "PhaseGate",
    "U1Gate",
    "U2Gate",
    "U3Gate",
    "UGate",
    "RXGate",
    "RYGate",
    "RZGate",
    "RotGate",
    "GlobalPhaseGate",
    # Gates — two-qubit
    "CNOTGate",
    "CXGate",
    "CZGate",
    "SwapGate",
    "ISwapGate",
    "ESCGate",
    "RXXGate",
    "RYYGate",
    "RZZGate",
    "RZXGate",
    "XYGate",
    "CRXGate",
    "CRYGate",
    "CRZGate",
    "CPhaseGate",
    "CSwapGate",
    "DCXGate",
    "ESCGate",
    "MSGate",
    # Gates — three-qubit
    "ToffoliGate",
    "CCXGate",
    "FredkinGate",
    "CCZGate",
    # Gates — multi-qubit
    "MCXGate",
    "MCZGate",
    # Gate helpers
    "GateLibrary",
    "controlled",
    "power",
    "dagger",
    # Operations
    "Operation",
    "CompositeOperation",
    "Barrier",
    "Reset",
    "ConditionalOperation",
    # Qubits
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
    # Registers
    "QuantumRegister",
    "ClassicalRegister",
    "Register",
    "RegisterSizeError",
    "RegisterIndexError",
    "RegisterConflictError",
    "registers_overlap",
    "total_register_size",
    "validate_register_names",
    # States
    "QuantumState",
    "Statevector",
    "DensityMatrix",
    "Operator",
    "Observable",
]
