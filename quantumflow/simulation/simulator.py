"""
Quantum Circuit Simulator Interface
=====================================

Provides the unified :class:`Simulator` abstract base class and three
concrete implementations:

* :class:`StatevectorSimulator` — ideal (noiseless) pure-state simulator
  based on :class:`~quantumflow.simulation.statevector.StatevectorBackend`.
* :class:`DensityMatrixSimulator` — mixed-state simulator based on
  :class:`~quantumflow.simulation.density_matrix.DensityMatrixBackend`,
  supporting noise channels and Kraus operators.
* :class:`MPSimulator` — Matrix Product State simulator for efficient
  simulation of shallow circuits on many qubits.

Also defines:

* :class:`SimulationResult` — a rich result object holding the final
  state, measurement counts, probabilities, individual-shot memory, and
  execution metadata.
* :class:`BackendConfig` — dataclass configuring max qubits, precision,
  device (cpu/gpu), seed, and optimisation level.
* :class:`SimulatorFactory` — convenience factory for creating simulators
  from configuration.

Typical usage::

    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.simulation.simulator import StatevectorSimulator

    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).measure([0, 1], [0, 1])

    sim = StatevectorSimulator()
    result = sim.run(qc, shots=1024)
    print(result.get_counts())
    result.plot_histogram()   # requires matplotlib
"""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import Gate, Measurement
from quantumflow.core.operation import Barrier, Operation, Reset
from quantumflow.core.state import DensityMatrix, Statevector

from quantumflow.simulation.statevector import StatevectorBackend
from quantumflow.simulation.density_matrix import DensityMatrixBackend

__all__ = [
    "Simulator",
    "StatevectorSimulator",
    "DensityMatrixSimulator",
    "MPSimulator",
    "SimulationResult",
    "BackendConfig",
    "SimulatorFactory",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLEX_DTYPE = np.complex128
_FLOAT_DTYPE = np.float64
_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# BackendConfig
# ---------------------------------------------------------------------------

@dataclass
class BackendConfig:
    """Configuration dataclass for simulator backends.

    Parameters
    ----------
    max_qubits : int
        Maximum number of qubits the simulator will handle.  Circuits
        exceeding this limit raise a :class:`ValueError`.
    precision : str
        ``'double'`` (complex128) or ``'single'`` (complex64).
    device : str
        ``'cpu'`` (default) or ``'gpu'`` (when CuPy is available).
    seed : int or None
        Random seed for reproducibility.
    optimization_level : int
        0 = no optimisation, 1 = gate fusion, 2 = full optimisation
        (gate fusion + circuit rewriting).  Currently only level 0 is
        fully implemented.
    """

    max_qubits: int = 30
    precision: str = "double"
    device: str = "cpu"
    seed: Optional[int] = None
    optimization_level: int = 0


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

class SimulationResult:
    """Result container returned by :meth:`Simulator.run`.

    Holds the final quantum state, measurement counts, probabilities,
    individual-shot memory, and execution metadata.

    Parameters
    ----------
    statevector : Statevector or numpy.ndarray or None
        Final quantum state (only for noiseless simulations).
    counts : dict or None
        Mapping from bit-string to count.
    probabilities : numpy.ndarray or None
        Outcome probabilities, shape ``(2ⁿ,)``.
    memory : list of str or None
        Individual shot results (bit-strings).
    metadata : dict
        Execution metadata (shots, time, device, etc.).
    density_matrix : DensityMatrix or numpy.ndarray or None
        Final density matrix (for noisy / mixed-state simulations).
    """

    def __init__(
        self,
        statevector: Optional[Union[Statevector, np.ndarray]] = None,
        counts: Optional[Dict[str, int]] = None,
        probabilities: Optional[np.ndarray] = None,
        memory: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        density_matrix: Optional[Union[DensityMatrix, np.ndarray]] = None,
    ) -> None:
        self._statevector = statevector
        self._counts = counts if counts is not None else {}
        self._probabilities = probabilities
        self._memory = memory if memory is not None else []
        self._metadata = metadata if metadata is not None else {}
        self._density_matrix = density_matrix

    # -- Properties ----------------------------------------------------------

    @property
    def statevector(self) -> Optional[Statevector]:
        """Statevector or None: Final quantum state (noiseless only)."""
        if self._statevector is None:
            return None
        if isinstance(self._statevector, Statevector):
            return self._statevector
        return Statevector(self._statevector, normalize=False)

    @property
    def density_matrix(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Final density matrix."""
        if isinstance(self._density_matrix, DensityMatrix):
            return self._density_matrix.data
        return self._density_matrix

    @property
    def counts(self) -> Dict[str, int]:
        """dict: Mapping from bit-string to count."""
        return dict(self._counts)

    @property
    def probabilities_array(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Outcome probabilities array."""
        return self._probabilities

    @property
    def memory(self) -> List[str]:
        """list of str: Individual shot results."""
        return list(self._memory)

    @property
    def metadata(self) -> Dict[str, Any]:
        """dict: Execution metadata."""
        return dict(self._metadata)

    @property
    def num_qubits(self) -> Optional[int]:
        """int or None: Number of qubits (inferred from results)."""
        if self._probabilities is not None:
            n = len(self._probabilities)
            if n > 0:
                return int(round(math.log2(n)))
        if self._counts:
            first_key = next(iter(self._counts))
            return len(first_key)
        if self._statevector is not None:
            if isinstance(self._statevector, Statevector):
                return self._statevector.num_qubits
            return int(round(math.log2(len(self._statevector))))
        if self._density_matrix is not None:
            dm = self._density_matrix
            if isinstance(dm, DensityMatrix):
                return dm.num_qubits
            return int(round(math.log2(dm.shape[0])))
        return None

    # -- Derived methods -----------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        """Return measurement counts as ``{bitstring: count}``.

        Returns
        -------
        dict
        """
        return dict(self._counts)

    def get_probabilities(self) -> Dict[str, float]:
        """Return outcome probabilities as ``{bitstring: probability}``.

        Returns
        -------
        dict
            Only entries with non-zero probability are included.
        """
        if self._probabilities is None:
            # Derive from counts
            total = sum(self._counts.values())
            if total == 0:
                return {}
            return {k: v / total for k, v in self._counts.items()}

        n_qubits = self.num_qubits
        if n_qubits is None:
            return {}
        result: Dict[str, float] = {}
        for i, p in enumerate(self._probabilities):
            if p > _TOLERANCE:
                bits = format(i, f"0{n_qubits}b")
                result[bits] = float(p)
        return result

    def get_memory(self) -> List[str]:
        """Return individual shot results.

        Returns
        -------
        list of str
        """
        return list(self._memory)

    def most_frequent(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return the *n* most frequent measurement outcomes.

        Parameters
        ----------
        n : int

        Returns
        -------
        list of (str, int)
            Sorted by count descending.
        """
        sorted_items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    # -- Visualization -------------------------------------------------------

    def plot_histogram(
        self,
        title: str = "Measurement Outcomes",
        figsize: Tuple[float, float] = (10, 6),
        color: str = "#4C72B0",
        ax: Any = None,
    ) -> Any:
        """Plot a bar chart of measurement counts.

        Requires ``matplotlib`` to be installed.

        Parameters
        ----------
        title : str
        figsize : tuple of float
        color : str
        ax : matplotlib Axes, optional
            Existing axes to plot on.

        Returns
        -------
        matplotlib Axes or None
            ``None`` if matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        except ImportError:
            warnings = __import__("warnings")
            warnings.warn(
                "matplotlib is required for plot_histogram(). "
                "Install it with: pip install matplotlib",
                stacklevel=2,
            )
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if not self._counts:
            ax.set_title(f"{title} (no counts)")
            return ax

        labels = list(self._counts.keys())
        values = list(self._counts.values())

        ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

        return ax

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        nq = self.num_qubits
        nc = len(self._counts)
        shots = self._metadata.get("shots", None)
        parts = [f"SimulationResult(num_qubits={nq}, outcomes={nc}"]
        if shots is not None:
            parts.append(f", shots={shots}")
        parts.append(")")
        return "".join(parts)

    def __str__(self) -> str:
        lines = [repr(self)]
        if self._counts:
            lines.append("  Counts:")
            for bitstring, count in sorted(self._counts.items()):
                lines.append(f"    {bitstring}: {count}")
        if self._metadata:
            lines.append("  Metadata:")
            for k, v in self._metadata.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract Simulator
# ---------------------------------------------------------------------------

class Simulator(abc.ABC):
    """Abstract base class for quantum circuit simulators.

    All concrete simulators must implement the methods defined here.
    """

    @abc.abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> SimulationResult:
        """Execute a circuit and return a :class:`SimulationResult`.

        Parameters
        ----------
        circuit : QuantumCircuit
        shots : int
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        SimulationResult
        """
        ...

    @abc.abstractmethod
    def run_batch(
        self,
        circuits: Sequence[QuantumCircuit],
        shots: int = 1024,
    ) -> List[SimulationResult]:
        """Execute a batch of circuits.

        Parameters
        ----------
        circuits : sequence of QuantumCircuit
        shots : int

        Returns
        -------
        list of SimulationResult
        """
        ...

    @abc.abstractmethod
    def expectation(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> float:
        r"""Compute ``⟨O⟩`` for the circuit output state.

        Parameters
        ----------
        circuit : QuantumCircuit
        observable : numpy.ndarray
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        float
        """
        ...

    @abc.abstractmethod
    def sample(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Dict[str, int]:
        """Sample measurement outcomes.

        Parameters
        ----------
        circuit : QuantumCircuit
        shots : int
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        dict
            ``{bitstring: count}``
        """
        ...

    @abc.abstractmethod
    def state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Union[Statevector, DensityMatrix, np.ndarray]:
        """Return the final quantum state without measuring.

        Parameters
        ----------
        circuit : QuantumCircuit
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        Statevector or DensityMatrix or numpy.ndarray
        """
        ...

    @abc.abstractmethod
    def probabilities(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        """Return outcome probabilities without measuring.

        Parameters
        ----------
        circuit : QuantumCircuit
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        numpy.ndarray
            Shape ``(2ⁿ,)``.
        """
        ...


# ---------------------------------------------------------------------------
# StatevectorSimulator
# ---------------------------------------------------------------------------

class StatevectorSimulator(Simulator):
    """Ideal (noiseless) pure-state simulator.

    Uses the :class:`StatevectorBackend` internally.

    Parameters
    ----------
    config : BackendConfig, optional
        Simulator configuration.

    Examples
    --------
    >>> sim = StatevectorSimulator()
    >>> result = sim.run(bell_circuit, shots=1024)
    >>> result.get_counts()
    {'00': ~512, '11': ~512}
    """

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        self._config = config or BackendConfig()
        self._backend = StatevectorBackend(
            precision=self._config.precision,
            seed=self._config.seed,
        )

    @property
    def config(self) -> BackendConfig:
        """BackendConfig: Current configuration."""
        return self._config

    @property
    def backend(self) -> StatevectorBackend:
        """StatevectorBackend: The underlying backend."""
        return self._backend

    def _check_qubits(self, circuit: QuantumCircuit) -> None:
        """Validate that the circuit does not exceed the configured
        maximum number of qubits."""
        if circuit.num_qubits > self._config.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"exceeding the configured maximum of "
                f"{self._config.max_qubits}"
            )

    def _resolve_initial_state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Convert the initial state to a raw numpy array."""
        if initial_state is None:
            return None
        if isinstance(initial_state, Statevector):
            return self._backend.from_statevector(initial_state)
        return np.asarray(initial_state, dtype=_COMPLEX_DTYPE)

    # -- Simulator interface -------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> SimulationResult:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)

        t0 = time.perf_counter()

        # Run the circuit
        final_state = self._backend.run_circuit(circuit, init)

        # Compute probabilities
        probs = self._backend.probabilities(final_state)

        # Sample if shots > 0
        counts: Dict[str, int] = {}
        memory: List[str] = []
        if shots > 0:
            counts = self._backend.sample(final_state, shots, circuit.num_qubits)
            # Build memory list
            all_shots: Dict[str, List[str]] = {}
            rng = np.random.default_rng(self._config.seed)
            for bitstring, count in counts.items():
                all_shots[bitstring] = [bitstring] * count
            memory_list: List[str] = []
            for bitstring, shots_list in all_shots.items():
                memory_list.extend(shots_list)
            # Shuffle for randomness
            rng.shuffle(memory_list)
            memory = memory_list[:shots]

        elapsed = time.perf_counter() - t0

        metadata = {
            "shots": shots,
            "time_seconds": elapsed,
            "device": self._config.device,
            "precision": self._config.precision,
            "simulator_type": "statevector",
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
        }

        return SimulationResult(
            statevector=final_state.copy(),
            counts=counts,
            probabilities=probs,
            memory=memory,
            metadata=metadata,
        )

    def run_batch(
        self,
        circuits: Sequence[QuantumCircuit],
        shots: int = 1024,
    ) -> List[SimulationResult]:
        return [self.run(c, shots=shots) for c in circuits]

    def expectation(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> float:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_state = self._backend.run_circuit(circuit, init)
        return self._backend.expectation_value(final_state, observable)

    def sample(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Dict[str, int]:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_state = self._backend.run_circuit(circuit, init)
        return self._backend.sample(final_state, shots, circuit.num_qubits)

    def state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Statevector:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_state = self._backend.run_circuit(circuit, init)
        return self._backend.statevector_from_array(final_state)

    def probabilities(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_state = self._backend.run_circuit(circuit, init)
        return self._backend.probabilities(final_state)

    def grad_params(
        self,
        circuit: QuantumCircuit,
        param_index: int,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute the gradient of the statevector w.r.t. a parameter.

        Parameters
        ----------
        circuit : QuantumCircuit
        param_index : int
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        numpy.ndarray
        """
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        return self._backend.grad_params(circuit, param_index, init)

    def expectation_grad(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        param_index: int,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> float:
        r"""Compute ``∂⟨O⟩/∂θᵢ``.

        Parameters
        ----------
        circuit : QuantumCircuit
        observable : numpy.ndarray
        param_index : int
        initial_state : Statevector or numpy.ndarray, optional

        Returns
        -------
        float
        """
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        return self._backend.expectation_grad(circuit, observable, param_index, init)

    def __repr__(self) -> str:
        return (
            f"StatevectorSimulator("
            f"max_qubits={self._config.max_qubits}, "
            f"precision={self._config.precision!r}, "
            f"device={self._config.device!r})"
        )


# ---------------------------------------------------------------------------
# DensityMatrixSimulator
# ---------------------------------------------------------------------------

class DensityMatrixSimulator(Simulator):
    """Mixed-state simulator with noise support.

    Uses the :class:`DensityMatrixBackend` internally and can optionally
    apply noise channels after each gate.

    Parameters
    ----------
    config : BackendConfig, optional
    noise_model : object, optional
        An object providing ``after_gate(rho, gate, qubits, num_qubits)``.
        If ``None`` the simulation is noiseless (but still uses density
        matrices).

    Examples
    --------
    >>> sim = DensityMatrixSimulator()
    >>> result = sim.run(circuit, shots=1024)
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        noise_model: Optional[Any] = None,
    ) -> None:
        self._config = config or BackendConfig()
        self._backend = DensityMatrixBackend(
            precision=self._config.precision,
            seed=self._config.seed,
        )
        self._noise_model = noise_model

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def backend(self) -> DensityMatrixBackend:
        return self._backend

    def _check_qubits(self, circuit: QuantumCircuit) -> None:
        if circuit.num_qubits > self._config.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"exceeding the configured maximum of "
                f"{self._config.max_qubits}"
            )

    def _resolve_initial_state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, DensityMatrix, np.ndarray]],
    ) -> Optional[np.ndarray]:
        if initial_state is None:
            return None
        if isinstance(initial_state, Statevector):
            return self._backend.from_statevector(initial_state)
        if isinstance(initial_state, DensityMatrix):
            return self._backend.from_density_matrix(initial_state)
        arr = np.asarray(initial_state, dtype=_COMPLEX_DTYPE)
        if arr.ndim == 1:
            return self._backend.from_statevector(arr)
        return self._backend.from_density_matrix(arr)

    # -- Simulator interface -------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> SimulationResult:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)

        t0 = time.perf_counter()
        final_rho = self._backend.run_circuit(
            circuit, init, noise_model=self._noise_model
        )

        probs = self._backend.probabilities(final_rho)

        counts: Dict[str, int] = {}
        memory: List[str] = []
        if shots > 0:
            counts = self._backend.sample(final_rho, shots, circuit.num_qubits)
            memory_list: List[str] = []
            for bitstring, count in counts.items():
                memory_list.extend([bitstring] * count)
            rng = np.random.default_rng(self._config.seed)
            rng.shuffle(memory_list)
            memory = memory_list[:shots]

        elapsed = time.perf_counter() - t0

        metadata = {
            "shots": shots,
            "time_seconds": elapsed,
            "device": self._config.device,
            "precision": self._config.precision,
            "simulator_type": "density_matrix",
            "noise": self._noise_model is not None,
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
        }

        return SimulationResult(
            density_matrix=final_rho.copy(),
            counts=counts,
            probabilities=probs,
            memory=memory,
            metadata=metadata,
        )

    def run_batch(
        self,
        circuits: Sequence[QuantumCircuit],
        shots: int = 1024,
    ) -> List[SimulationResult]:
        return [self.run(c, shots=shots) for c in circuits]

    def expectation(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> float:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_rho = self._backend.run_circuit(
            circuit, init, noise_model=self._noise_model
        )
        return self._backend.expectation_value(final_rho, observable)

    def sample(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Dict[str, int]:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_rho = self._backend.run_circuit(
            circuit, init, noise_model=self._noise_model
        )
        return self._backend.sample(final_rho, shots, circuit.num_qubits)

    def state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_rho = self._backend.run_circuit(
            circuit, init, noise_model=self._noise_model
        )
        return final_rho

    def probabilities(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_qubits(circuit)
        init = self._resolve_initial_state(circuit, initial_state)
        final_rho = self._backend.run_circuit(
            circuit, init, noise_model=self._noise_model
        )
        return self._backend.probabilities(final_rho)

    def __repr__(self) -> str:
        return (
            f"DensityMatrixSimulator("
            f"max_qubits={self._config.max_qubits}, "
            f"precision={self._config.precision!r}, "
            f"noise={'yes' if self._noise_model else 'no'})"
        )


# ---------------------------------------------------------------------------
# MPSimulator — Matrix Product State
# ---------------------------------------------------------------------------

class _MPS:
    """Internal representation of a Matrix Product State.

    An MPS for *n* qubits is stored as a list of *n* tensors, where
    tensor ``i`` has shape ``(bond_dim_left, 2, bond_dim_right)``.
    The leftmost tensor has ``bond_dim_left = 1`` and the rightmost
    has ``bond_dim_right = 1``.

    Parameters
    ----------
    tensors : list of numpy.ndarray
        The MPS tensors.
    """

    def __init__(self, tensors: List[np.ndarray]) -> None:
        self.tensors = tensors

    @property
    def num_qubits(self) -> int:
        return len(self.tensors)

    def copy(self) -> _MPS:
        return _MPS([t.copy() for t in self.tensors])


def _mps_zero_state(num_qubits: int) -> _MPS:
    """Create the ``|00…0⟩`` state as an MPS."""
    tensors: List[np.ndarray] = []
    for i in range(num_qubits):
        if i == 0 and num_qubits == 1:
            # Single qubit: shape (1, 2, 1)
            t = np.zeros((1, 2, 1), dtype=_COMPLEX_DTYPE)
            t[0, 0, 0] = 1.0
        elif i == 0:
            # First qubit: shape (1, 2, 1)
            t = np.zeros((1, 2, 1), dtype=_COMPLEX_DTYPE)
            t[0, 0, 0] = 1.0
        elif i == num_qubits - 1:
            # Last qubit: shape (1, 2, 1)
            t = np.zeros((1, 2, 1), dtype=_COMPLEX_DTYPE)
            t[0, 0, 0] = 1.0
        else:
            # Middle qubit: shape (1, 2, 1)
            t = np.zeros((1, 2, 1), dtype=_COMPLEX_DTYPE)
            t[0, 0, 0] = 1.0
        tensors.append(t)
    return _MPS(tensors)


def _apply_gate_mps(
    mps: _MPS,
    gate_matrix: np.ndarray,
    qubits: Sequence[int],
    max_bond: int = 64,
) -> _MPS:
    """Apply a gate to specific qubits of an MPS.

    For single-qubit gates this contracts the gate with the
    corresponding MPS tensor directly.  For two-qubit gates the two
    adjacent (or non-adjacent) tensors are contracted, the gate is
    applied, and the result is split back via SVD with bond-dimension
    truncation.

    Parameters
    ----------
    mps : _MPS
    gate_matrix : numpy.ndarray
        Shape ``(2**k, 2**k)``.
    qubits : sequence of int
    max_bond : int
        Maximum bond dimension after truncation.

    Returns
    -------
    _MPS
    """
    qubits = sorted(qubits)
    k = len(qubits)
    tensors = mps.copy().tensors
    n = mps.num_qubits

    if k == 1:
        q = qubits[0]
        # Contract gate with single tensor
        t = tensors[q]  # (d_left, 2, d_right)
        d_left = t.shape[0]
        d_right = t.shape[2]

        # Reshape: (d_left, 2, d_right) -> (d_left * 2, d_right)
        t_reshaped = t.reshape(d_left * 2, d_right)
        # Apply gate: (2, 2) acts on the physical index
        # We need: new_t[a, i, b] = sum_j U[i,j] * t[a, j, b]
        # Use einsum
        new_t = np.einsum("ij,ajb->aib", gate_matrix, t)
        new_t = np.asarray(new_t)
        tensors[q] = new_t
        return _MPS(tensors)

    if k == 2:
        q1, q2 = qubits

        if q2 == q1 + 1:
            # Adjacent qubits — direct SVD approach
            tensors = list(tensors)
            eff_q1 = q1
            eff_q2 = q2

            t1 = tensors[eff_q1]  # (d_left, 2, d_mid)
            t2 = tensors[eff_q2]  # (d_mid, 2, d_right)
            d_left = t1.shape[0]
            d_right = t2.shape[2]

            theta = np.einsum("ais,sjb->aijb", t1, t2)
            theta_4 = theta.reshape(d_left, 4, d_right)
            gate_4 = gate_matrix.reshape(4, 4)
            theta_gate = np.einsum("ij,ajb->aib", gate_4, theta_4)
            theta_gate = np.asarray(theta_gate)

            left_mat = theta_gate.reshape(d_left * 2, 2 * d_right)
            U, S, Vh = np.linalg.svd(left_mat, full_matrices=False)
            if len(S) > max_bond:
                U = U[:, :max_bond]
                S = S[:max_bond]
                Vh = Vh[:max_bond, :]

            bond = len(S)
            t1_new = U.reshape(d_left, 2, bond)
            t2_new = (np.diag(S) @ Vh).reshape(bond, 2, d_right)

            tensors[eff_q1] = t1_new
            tensors[eff_q2] = t2_new
            return _MPS(tensors)
        else:
            # Non-adjacent qubits — fall back to full statevector
            # to guarantee correctness
            return _apply_gate_mps_full(mps, gate_matrix, qubits, max_bond)

    # General multi-qubit gate: decompose into two-qubit gates
    # For simplicity, use full contraction for small k
    if k <= 4:
        return _apply_gate_mps_full(mps, gate_matrix, qubits, max_bond)

    raise NotImplementedError(
        f"MPS gate application for {k}-qubit gates not yet implemented"
    )


def _apply_gate_mps_full(
    mps: _MPS,
    gate_matrix: np.ndarray,
    qubits: Sequence[int],
    max_bond: int = 64,
) -> _MPS:
    """Apply a multi-qubit gate by contracting to a full statevector
    and then converting back to MPS.

    Uses the einsum-based gate application from StatevectorBackend
    for correctness on arbitrary qubit layouts.
    """
    sv = mps_to_statevector(mps)
    n = mps.num_qubits

    # Apply gate using the correct einsum-based approach
    from quantumflow.simulation.statevector import StatevectorBackend
    backend = StatevectorBackend()
    backend.apply_gate_full(sv, gate_matrix, qubits, n)

    norm = np.linalg.norm(sv)
    if norm > _TOLERANCE:
        sv /= norm

    return statevector_to_mps(sv, n, max_bond)


def _embed_operator_simple(
    operator: np.ndarray,
    qubits: Sequence[int],
    n: int,
) -> np.ndarray:
    """Embed operator on target qubits into full Hilbert space.
    Simplified version for small systems."""
    k = len(qubits)
    dim = 1 << n
    if k == n:
        return operator

    perm = list(qubits) + [q for q in range(n) if q not in qubits]
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    gate_on_front = np.kron(operator, np.eye(1 << (n - k), dtype=_COMPLEX_DTYPE))
    P = _build_perm_matrix(perm, n)
    P_inv = _build_perm_matrix(inv_perm, n)
    return P_inv @ gate_on_front @ P


def _build_perm_matrix(perm: List[int], n: int) -> np.ndarray:
    """Build permutation matrix for qubit reordering."""
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


def _mps_swap_qubits(
    tensors: List[np.ndarray],
    q1: int,
    q2: int,
) -> List[np.ndarray]:
    """Swap two adjacent MPS tensors using SWAP-like SVD manipulation.

    For non-adjacent qubits, perform a sequence of adjacent swaps.

    Parameters
    ----------
    tensors : list of numpy.ndarray
    q1 : int
    q2 : int

    Returns
    -------
    list of numpy.ndarray
    """
    result = list(tensors)
    a, b = min(q1, q2), max(q1, q2)
    while a < b:
        result = _mps_swap_adjacent(result, a)
        a += 1
    return result


def _mps_swap_adjacent(
    tensors: List[np.ndarray],
    i: int,
) -> List[np.ndarray]:
    """Swap two adjacent MPS tensors at positions *i* and *i+1*.

    Implements the SWAP gate on adjacent qubits using SVD.

    Parameters
    ----------
    tensors : list of numpy.ndarray
    i : int

    Returns
    -------
    list of numpy.ndarray
    """
    if i + 1 >= len(tensors):
        return tensors

    t1 = tensors[i]  # (dL, 2, dM)
    t2 = tensors[i + 1]  # (dM, 2, dR)

    dL = t1.shape[0]
    dR = t2.shape[2]
    dM = t1.shape[2]

    # theta[a, i, j, b] = t1[a, i, s] * t2[s, j, b]
    theta = np.einsum("ais,sjb->aijb", t1, t2)

    # Apply SWAP: swap the physical indices i and j
    # SWAP matrix: (0,0)->(0,0), (0,1)->(1,0), (1,0)->(0,1), (1,1)->(1,1)
    theta_swapped = np.transpose(theta, (0, 2, 1, 3))  # (a, j, i, b)

    # Split back via SVD
    theta_mat = theta_swapped.reshape(dL * 2, 2 * dR)
    U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    bond = len(S)
    t1_new = U.reshape(dL, 2, bond)
    S_mat = np.diag(S)
    t2_new = (S_mat @ Vh).reshape(bond, 2, dR)

    result = list(tensors)
    result[i] = t1_new
    result[i + 1] = t2_new
    return result


def mps_to_statevector(mps: _MPS) -> np.ndarray:
    """Contract an MPS to a full statevector.

    Parameters
    ----------
    mps : _MPS

    Returns
    -------
    numpy.ndarray
        Shape ``(2ⁿ,)``.
    """
    tensor = mps.tensors[0]
    for i in range(1, mps.num_qubits):
        # tensor: (..., d_i), next: (d_i, 2, d_{i+1})
        tensor = np.tensordot(tensor, mps.tensors[i], axes=(-1, 0))
    # Final shape: (2, 2, ..., 2) -> flatten
    return tensor.reshape(-1)


def statevector_to_mps(
    state: np.ndarray,
    num_qubits: int,
    max_bond: int = 64,
) -> _MPS:
    """Convert a statevector to MPS form via successive SVDs.

    Parameters
    ----------
    state : numpy.ndarray
        Shape ``(2ⁿ,)``.
    num_qubits : int
    max_bond : int

    Returns
    -------
    _MPS
    """
    n = num_qubits
    if n == 0:
        return _MPS([])

    # Reshape to (1, 2, 2, ..., 2) — prepend trivial bond dimension
    tensor = state.reshape([1] + [2] * n).copy()

    tensors: List[np.ndarray] = []

    for i in range(n - 1):
        # tensor shape: (d_bond_left, 2, 2, ..., 2)  with bond + (n-i) dims
        d_bond_left = tensor.shape[0]
        remaining_physical = n - i - 1  # number of physical dims after this split
        flat_right = 1 << remaining_physical  # 2^(n-i-1)

        # Reshape to (d_bond_left * 2, flat_right) and SVD
        matrix = tensor.reshape(d_bond_left * 2, flat_right)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

        # Truncate
        if len(S) > max_bond:
            U = U[:, :max_bond]
            S = S[:max_bond]
            Vh = Vh[:max_bond, :]

        bond = len(S)

        # Store the left tensor: (d_bond_left, 2, bond)
        if i == 0:
            tensors.append(U.reshape(1, 2, bond))
        else:
            tensors.append(U.reshape(d_bond_left, 2, bond))

        # Update tensor: diag(S) @ Vh reshaped with remaining physical dims
        # (bond, flat_right) -> (bond, 2, 2, ..., 2)
        tensor = (np.diag(S) @ Vh).reshape(bond, *([2] * remaining_physical))

    # Last tensor: (bond, 2) -> (bond, 2, 1)
    d_bond_left = tensor.shape[0]
    tensors.append(tensor.reshape(d_bond_left, 2, 1))

    return _MPS(tensors)


# ---------------------------------------------------------------------------
# MPSimulator
# ---------------------------------------------------------------------------

class MPSimulator(Simulator):
    """Matrix Product State simulator for efficient simulation of
    shallow circuits on many qubits.

    Stores the quantum state as an MPS with configurable maximum bond
    dimension, enabling simulation of circuits that would be intractable
    with full statevector methods.

    Parameters
    ----------
    config : BackendConfig, optional
    max_bond_dimension : int, optional
        Maximum bond dimension for the MPS.  Default 64.

    Examples
    --------
    >>> sim = MPSimulator(max_bond_dimension=32)
    >>> result = sim.run(circuit, shots=1024)
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        max_bond_dimension: int = 64,
    ) -> None:
        self._config = config or BackendConfig()
        self._max_bond = max_bond_dimension
        self._rng = np.random.default_rng(self._config.seed)
        self._gate_cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = {}

    @property
    def config(self) -> BackendConfig:
        return self._config

    @property
    def max_bond_dimension(self) -> int:
        return self._max_bond

    def _get_gate_matrix(self, gate: Gate, params: Tuple[float, ...] = ()) -> np.ndarray:
        cache_key = (gate.name, params)
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]
        if params:
            mat = gate.to_matrix(*params)
        else:
            mat = gate.matrix
        self._gate_cache[cache_key] = mat
        return mat

    def _check_qubits(self, circuit: QuantumCircuit) -> None:
        if circuit.num_qubits > self._config.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"exceeding the configured maximum of "
                f"{self._config.max_qubits}"
            )

    # -- Simulator interface -------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> SimulationResult:
        self._check_qubits(circuit)
        n = circuit.num_qubits

        t0 = time.perf_counter()

        # Build MPS
        if initial_state is not None:
            if isinstance(initial_state, Statevector):
                sv = initial_state.data
            else:
                sv = np.asarray(initial_state, dtype=_COMPLEX_DTYPE)
            mps = statevector_to_mps(sv, n, self._max_bond)
        else:
            mps = _mps_zero_state(n)

        # Apply gates
        for op in circuit.data:
            if isinstance(op, Barrier):
                continue
            if isinstance(op, Reset):
                mps = self._apply_reset_mps(mps, op.qubits, n)
                continue
            if isinstance(op, Operation):
                if isinstance(op.gate, Measurement):
                    mps = self._apply_measurement_mps(mps, op.qubits, n)
                    continue
                gate_mat = self._get_gate_matrix(op.gate, op.params)
                mps = _apply_gate_mps(mps, gate_mat, op.qubits, self._max_bond)
                continue

        # Contract to full statevector for sampling
        final_sv = mps_to_statevector(mps)
        norm = np.linalg.norm(final_sv)
        if norm > _TOLERANCE:
            final_sv /= norm

        probs = np.real(np.abs(final_sv) ** 2)
        probs_total = probs.sum()
        if probs_total > _TOLERANCE:
            probs /= probs_total
        else:
            probs = np.ones_like(probs) / len(probs)

        counts: Dict[str, int] = {}
        memory: List[str] = []
        if shots > 0:
            outcomes = self._rng.choice(len(probs), size=shots, p=probs)
            for o in outcomes:
                bits = format(int(o), f"0{n}b")
                counts[bits] = counts.get(bits, 0) + 1
            for bitstring, count in counts.items():
                memory.extend([bitstring] * count)
            self._rng.shuffle(memory)
            memory = memory[:shots]

        elapsed = time.perf_counter() - t0

        metadata = {
            "shots": shots,
            "time_seconds": elapsed,
            "device": self._config.device,
            "precision": self._config.precision,
            "simulator_type": "mps",
            "max_bond_dimension": self._max_bond,
            "num_qubits": n,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
        }

        return SimulationResult(
            statevector=final_sv.copy(),
            counts=counts,
            probabilities=probs,
            memory=memory,
            metadata=metadata,
        )

    def run_batch(
        self,
        circuits: Sequence[QuantumCircuit],
        shots: int = 1024,
    ) -> List[SimulationResult]:
        return [self.run(c, shots=shots) for c in circuits]

    def expectation(
        self,
        circuit: QuantumCircuit,
        observable: np.ndarray,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> float:
        result = self.run(circuit, shots=0, initial_state=initial_state)
        sv = result.statevector
        if sv is None:
            raise RuntimeError("No statevector available")
        O = np.asarray(observable, dtype=_COMPLEX_DTYPE)
        return float(np.real(np.vdot(sv.data, O @ sv.data)))

    def sample(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Dict[str, int]:
        result = self.run(circuit, shots=shots, initial_state=initial_state)
        return result.get_counts()

    def state(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> Statevector:
        result = self.run(circuit, shots=0, initial_state=initial_state)
        sv = result.statevector
        if sv is None:
            raise RuntimeError("No statevector available")
        return sv

    def probabilities(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[Union[Statevector, np.ndarray]] = None,
    ) -> np.ndarray:
        result = self.run(circuit, shots=0, initial_state=initial_state)
        p = result.probabilities_array
        if p is None:
            raise RuntimeError("No probabilities available")
        return p

    # -- MPS-specific operations ---------------------------------------------

    def _apply_reset_mps(
        self,
        mps: _MPS,
        qubits: Tuple[int, ...],
        n: int,
    ) -> _MPS:
        """Reset qubits in an MPS to |0⟩.

        For each qubit, replace its tensor with the |0⟩-projected
        tensor and normalise via SVD.

        Parameters
        ----------
        mps : _MPS
        qubits : tuple of int
        n : int

        Returns
        -------
        _MPS
        """
        tensors = mps.copy().tensors
        for q in qubits:
            t = tensors[q]  # (dL, 2, dR)
            dL = t.shape[0]
            dR = t.shape[2]
            # Zero out the |1⟩ component
            t_zero = t.copy()
            t_zero[:, 1, :] = 0.0
            # Normalise
            norm = np.linalg.norm(t_zero)
            if norm > _TOLERANCE:
                t_zero /= norm
            tensors[q] = t_zero
        return _MPS(tensors)

    def _apply_measurement_mps(
        self,
        mps: _MPS,
        qubits: Tuple[int, ...],
        n: int,
    ) -> _MPS:
        """Measure qubits in an MPS and collapse.

        Converts to statevector, measures, and converts back.

        Parameters
        ----------
        mps : _MPS
        qubits : tuple of int
        n : int

        Returns
        -------
        _MPS
        """
        sv = mps_to_statevector(mps)
        norm = np.linalg.norm(sv)
        if norm > _TOLERANCE:
            sv /= norm

        # Measure
        probs = np.real(np.abs(sv) ** 2)
        probs_total = probs.sum()
        if probs_total > _TOLERANCE:
            probs /= probs_total

        k = len(qubits)
        outcome = int(self._rng.choice(len(probs), p=probs))

        # Collapse
        outcome_bits = [(outcome >> (k - 1 - i)) & 1 for i in range(k)]
        mask = np.ones(len(sv), dtype=bool)
        dim = len(sv)
        for qi, bit in zip(qubits, outcome_bits):
            qmask = np.array(
                [((idx >> (n - 1 - qi)) & 1) == bit for idx in range(dim)],
                dtype=bool,
            )
            mask &= qmask
        sv[~mask] = 0.0
        norm = np.linalg.norm(sv)
        if norm > _TOLERANCE:
            sv /= norm

        return statevector_to_mps(sv, n, self._max_bond)

    def __repr__(self) -> str:
        return (
            f"MPSimulator("
            f"max_qubits={self._config.max_qubits}, "
            f"max_bond={self._max_bond})"
        )


# ---------------------------------------------------------------------------
# SimulatorFactory
# ---------------------------------------------------------------------------

class SimulatorFactory:
    """Factory for creating simulator instances from configuration.

    Examples
    --------
    >>> factory = SimulatorFactory()
    >>> sim = factory.create("statevector", max_qubits=20, seed=42)
    >>> sim = factory.create("density_matrix", seed=42)
    >>> sim = factory.create("mps", max_bond_dimension=32)
    """

    _registry: Dict[str, Type[Simulator]] = {
        "statevector": StatevectorSimulator,
        "density_matrix": DensityMatrixSimulator,
        "mps": MPSimulator,
    }

    @classmethod
    def register(
        cls,
        name: str,
        simulator_class: Type[Simulator],
    ) -> None:
        """Register a custom simulator class under *name*.

        Parameters
        ----------
        name : str
            Identifier used in :meth:`create`.
        simulator_class : type
            A subclass of :class:`Simulator`.
        """
        if not issubclass(simulator_class, Simulator):
            raise TypeError(
                f"{simulator_class} must be a subclass of Simulator"
            )
        cls._registry[name] = simulator_class

    @classmethod
    def available(cls) -> List[str]:
        """Return the names of all registered simulators.

        Returns
        -------
        list of str
        """
        return sorted(cls._registry.keys())

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> Simulator:
        """Create a simulator instance.

        Parameters
        ----------
        name : str
            Simulator name (e.g. ``'statevector'``, ``'density_matrix'``,
            ``'mps'``).
        **kwargs
            Forwarded to the simulator constructor.  Common options:

            * ``max_qubits`` (int) — maximum qubit count.
            * ``precision`` (str) — ``'double'`` or ``'single'``.
            * ``seed`` (int or None) — random seed.
            * ``max_bond_dimension`` (int) — for MPS only.

        Returns
        -------
        Simulator

        Raises
        ------
        ValueError
            If *name* is not a registered simulator.
        """
        name_lower = name.lower().replace("-", "_")
        if name_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown simulator '{name}'. Available: {available}"
            )

        # Extract MPS-specific args
        config_kwargs = {}
        mps_kwargs = {}
        mps_specific = {"max_bond_dimension"}
        for k, v in kwargs.items():
            if k in mps_specific:
                mps_kwargs[k] = v
            elif k in BackendConfig.__dataclass_fields__:
                config_kwargs[k] = v
            else:
                mps_kwargs[k] = v

        config = BackendConfig(**config_kwargs)
        sim_class = cls._registry[name_lower]

        if name_lower == "mps":
            return sim_class(config=config, **mps_kwargs)
        elif name_lower == "density_matrix":
            noise = kwargs.get("noise_model")
            return sim_class(config=config, noise_model=noise)
        else:
            return sim_class(config=config)
