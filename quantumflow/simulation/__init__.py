"""
QuantumFlow Simulation Engine
==============================

Provides high-performance quantum circuit simulation backends and a unified
``Simulator`` interface.

Submodules
----------
* :mod:`quantumflow.simulation.statevector` — full statevector simulation.
* :mod:`quantumflow.simulation.density_matrix` — density matrix simulation
  with noise / Kraus operator support.
* :mod:`quantumflow.simulation.simulator` — unified ``Simulator`` ABC,
  concrete implementations (``StatevectorSimulator``,
  ``DensityMatrixSimulator``, ``MPSimulator``), ``SimulationResult``, and
  ``BackendConfig``.

Quick start::

    from quantumflow.simulation import StatevectorSimulator, SimulationResult

    sim = StatevectorSimulator()
    result = sim.run(circuit, shots=1024)
    print(result.get_counts())
"""

from quantumflow.simulation.statevector import StatevectorBackend
from quantumflow.simulation.density_matrix import DensityMatrixBackend
from quantumflow.simulation.simulator import (
    BackendConfig,
    MPSimulator,
    Simulator,
    SimulatorFactory,
    SimulationResult,
    StatevectorSimulator,
    DensityMatrixSimulator,
)

__all__ = [
    # Backends
    "StatevectorBackend",
    "DensityMatrixBackend",
    # Simulator interface & implementations
    "Simulator",
    "StatevectorSimulator",
    "DensityMatrixSimulator",
    "MPSimulator",
    # Result & configuration
    "SimulationResult",
    "BackendConfig",
    "SimulatorFactory",
]
