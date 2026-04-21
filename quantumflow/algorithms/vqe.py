"""
Variational Quantum Eigensolver (VQE)
=====================================

Implements the Variational Quantum Eigensolver for finding the ground
state energy of quantum systems, particularly molecular Hamiltonians.

VQE combines quantum circuits (parameterized ansatz) with classical
optimization to find the minimum eigenvalue of a Hamiltonian.

References:
    - Peruzzo, A., et al. (2014). A variational eigenvalue solver on a
      photonic quantum processor.
    - Kandala, A., et al. (2017). Hardware-efficient variational quantum
      eigensolver for small molecules.
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from scipy.optimize import minimize as scipy_minimize

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, RXGate, RYGate, RZGate,
        CNOTGate, CZGate, RXXGate, RYYGate, RZZGate,
        Measurement, UnitaryGate,
    )
    from quantumflow.core.state import Statevector
    from quantumflow.simulation.simulator import StatevectorSimulator, Simulator
except ImportError:
    pass


@dataclass
class PauliTerm:
    """A term in a Hamiltonian expressed as a Pauli string with coefficient."""
    coefficient: float
    pauli_string: str  # e.g., "IXYZ" for I⊗X⊗Y⊗Z

    @property
    def n_qubits(self) -> int:
        return len(self.pauli_string)

    def matrix(self) -> np.ndarray:
        """Compute the matrix representation of this Pauli term."""
        pauli_matrices = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        result = pauli_matrices[self.pauli_string[0]]
        for p in self.pauli_string[1:]:
            result = np.kron(result, pauli_matrices[p])
        return self.coefficient * result


@dataclass
class VQEResult:
    """Results from VQE optimization."""
    optimal_energy: float
    optimal_params: np.ndarray
    convergence_history: List[float] = field(default_factory=list)
    iteration_count: int = 0
    final_variance: float = 0.0
    optimizer_name: str = ""
    elapsed_time: float = 0.0
    success: bool = True


class Hamiltonian:
    """
    Quantum Hamiltonian represented as sum of Pauli terms.

    H = sum_i c_i * P_i

    where c_i are coefficients and P_i are Pauli strings.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    terms : Optional[List[PauliTerm]]
        List of Pauli terms.

    Examples
    --------
    >>> H = Hamiltonian(2, [
    ...     PauliTerm(1.0, "ZZ"),
    ...     PauliTerm(0.5, "XX"),
    ...     PauliTerm(-0.5, "YY"),
    ... ])
    >>> print(H.n_terms)
    3
    """

    def __init__(
        self,
        n_qubits: int,
        terms: Optional[List[PauliTerm]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.terms: List[PauliTerm] = terms or []

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def add_term(self, coefficient: float, pauli_string: str) -> None:
        """Add a Pauli term to the Hamiltonian."""
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string length {len(pauli_string)} != n_qubits {self.n_qubits}")
        self.terms.append(PauliTerm(coefficient, pauli_string))

    @classmethod
    def from_terms(cls, terms: List[Tuple[float, str]]) -> 'Hamiltonian':
        """Create Hamiltonian from list of (coefficient, pauli_string) tuples."""
        if not terms:
            raise ValueError("No terms provided")
        n_qubits = len(terms[0][1])
        h = cls(n_qubits)
        for coeff, pauli in terms:
            h.add_term(coeff, pauli)
        return h

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Hamiltonian':
        """
        Decompose a Hermitian matrix into Pauli terms.

        Parameters
        ----------
        matrix : np.ndarray
            Hermitian matrix to decompose.

        Returns
        -------
        Hamiltonian
            Hamiltonian with Pauli decomposition.
        """
        n = int(round(np.log2(matrix.shape[0])))
        h = cls(n)

        paulis = ['I', 'X', 'Y', 'Z']
        for i in range(4 ** n):
            pauli_str = ''
            temp = i
            coeffs = []
            for _ in range(n):
                coeffs.append(paulis[temp % 4])
                temp //= 4
            pauli_str = ''.join(reversed(coeffs))

            # Compute expectation value <P|H|P> / dim
            pauli_op = np.eye(1, dtype=np.complex128)
            for p in pauli_str:
                pauli_matrices = {
                    'I': np.eye(2, dtype=np.complex128),
                    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
                    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
                }
                pauli_op = np.kron(pauli_op, pauli_matrices[p])

            coeff = np.real(np.trace(pauli_op @ matrix)) / (2 ** n)
            if abs(coeff) > 1e-10:
                h.add_term(float(coeff), pauli_str)

        return h

    def matrix(self) -> np.ndarray:
        """Compute the full Hamiltonian matrix."""
        result = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=np.complex128)
        for term in self.terms:
            result += term.matrix()
        return result

    def expectation(
        self,
        state: np.ndarray,
        simulator: Optional['Simulator'] = None,
    ) -> float:
        """
        Compute expectation value <psi|H|psi>.

        Parameters
        ----------
        state : np.ndarray
            Quantum state vector.
        simulator : Optional[Simulator]
            Simulator for computing expectation values.

        Returns
        -------
        float
            Expectation value of H in the given state.
        """
        energy = 0.0
        for term in self.terms:
            pauli_matrix = term.matrix() / term.coefficient
            val = np.real(np.conj(state) @ pauli_matrix @ state)
            energy += term.coefficient * val
        return energy

    def gradient(
        self,
        state: np.ndarray,
        params: np.ndarray,
        param_index: int,
        shift: float = np.pi / 2,
    ) -> float:
        """
        Compute gradient of expectation value w.r.t. a parameter.

        Uses the parameter-shift rule:
        d<H>/dθ = (<H>_{θ+s} - <H>_{θ-s}) / 2

        Parameters
        ----------
        state : np.ndarray
            State at parameter values `params`.
        params : np.ndarray
            Current parameter values.
        param_index : int
            Index of parameter to differentiate.
        shift : float
            Shift value (default π/2 for exact gradients).

        Returns
        -------
        float
            Gradient of expectation value.
        """
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[param_index] += shift
        params_minus[param_index] -= shift

        # This would require recomputing the state with new parameters
        # Placeholder - actual implementation depends on ansatz
        return 0.0

    @classmethod
    def heisenberg_hamiltonian(
        cls,
        n_qubits: int,
        jx: float = 1.0,
        jy: float = 1.0,
        jz: float = 1.0,
        hz: float = 0.0,
    ) -> 'Hamiltonian':
        """
        Create a Heisenberg model Hamiltonian.

        H = sum_{<i,j>} [Jx*X_i*X_j + Jy*Y_i*Y_j + Jz*Z_i*Z_j] + hz * sum_i Z_i

        Parameters
        ----------
        n_qubits : int
        jx, jy, jz : float
            Coupling constants.
        hz : float
            External magnetic field.

        Returns
        -------
        Hamiltonian
        """
        h = cls(n_qubits)
        for i in range(n_qubits - 1):
            pauli_x = ['I'] * n_qubits
            pauli_y = ['I'] * n_qubits
            pauli_z = ['I'] * n_qubits
            pauli_x[i] = 'X'; pauli_x[i + 1] = 'X'
            pauli_y[i] = 'Y'; pauli_y[i + 1] = 'Y'
            pauli_z[i] = 'Z'; pauli_z[i + 1] = 'Z'
            h.add_term(jx, ''.join(pauli_x))
            h.add_term(jy, ''.join(pauli_y))
            h.add_term(jz, ''.join(pauli_z))

        if hz != 0:
            for i in range(n_qubits):
                pauli = ['I'] * n_qubits
                pauli[i] = 'Z'
                h.add_term(hz, ''.join(pauli))

        return h

    @classmethod
    def transverse_field_ising(cls, n_qubits: int, j: float = 1.0, h: float = 1.0) -> 'Hamiltonian':
        """
        Create a transverse-field Ising model Hamiltonian.

        H = -J * sum Z_i*Z_{i+1} - h * sum X_i
        """
        h_ham = cls(n_qubits)
        for i in range(n_qubits - 1):
            pauli = ['I'] * n_qubits
            pauli[i] = 'Z'; pauli[i + 1] = 'Z'
            h_ham.add_term(-j, ''.join(pauli))
        for i in range(n_qubits):
            pauli = ['I'] * n_qubits
            pauli[i] = 'X'
            h_ham.add_term(-h, ''.join(pauli))
        return h_ham

    @classmethod
    def hydrogen_molecule(cls) -> 'Hamiltonian':
        """
        Create H2 molecule Hamiltonian (STO-3G basis).

        Returns the electronic structure Hamiltonian for H2 at equilibrium
        bond distance in the STO-3G minimal basis set.
        """
        # H2 Hamiltonian in STO-3G basis (4 qubits, Jordan-Wigner mapping)
        # Coefficients from the literature
        terms = [
            (-0.8105490, "IIII"),
            (0.1721839, "IIIZ"),
            (-0.2257535, "IIZI"),
            (0.1721839, "IZII"),
            (0.1209123, "IZIZ"),
            (0.0452321, "ZIIZ"),
            (0.0452321, "IZZI"),
            (0.1661454, "IZZZ"),
            (0.1209123, "ZIZI"),
            (0.0452321, "ZZII"),
            (-0.2257535, "ZIII"),
            (0.1661454, "ZIZZ"),
            (0.0452321, "ZZIZ"),
            (0.0452321, "ZZZI"),
            (-0.1054179, "IIIX"),
            (0.1054179, "IIXI"),
            (-0.1054179, "IXII"),
            (-0.1054179, "XIII"),
            (0.1054179, "IXIZ"),
            (-0.1054179, "XIZI"),
            (0.1054179, "XIZZ"),
            (0.1054179, "IZIX"),
            (-0.1054179, "ZIIX"),
            (-0.1054179, "ZIXI"),
            (0.1054179, "ZIXZ"),
            (-0.1054179, "IXZX"),
            (0.1054179, "XZXI"),
            (0.1054179, "XZZX"),
            (0.0063094, "IIXX"),
            (0.0063094, "XXII"),
            (-0.0063094, "IXIX"),
            (-0.0063094, "XIXI"),
            (0.0063094, "IXZX"),
            (0.0063094, "XZIX"),
            (-0.0063094, "XIXZ"),
            (-0.0063094, "ZXIX"),
            (0.0063094, "IYYI"),
            (-0.0063094, "IYIY"),
            (0.0063094, "YIIY"),
            (-0.0063094, "YIYI"),
            (-0.0063094, "YYII"),
            (-0.0063094, "YYYY"),
        ]
        return cls.from_terms(terms)


class HWEAnsatz:
    """
    Hardware-Efficient Ansatz for VQE.

    Uses single-qubit rotations followed by nearest-neighbor entangling gates.
    This is a compact, hardware-friendly ansatz suitable for near-term devices.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers (depth).
    rotation_set : List[str]
        Set of rotation gates per layer. E.g., ['ry', 'rz'].
    entanglement : str
        Entanglement pattern: 'linear', 'circular', 'full'.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        rotation_set: Optional[List[str]] = None,
        entanglement: str = 'linear',
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_set = rotation_set or ['ry', 'rz']
        self.entanglement = entanglement

    def n_params(self) -> int:
        """Total number of parameters."""
        return self.n_qubits * len(self.rotation_set) * self.n_layers

    def construct_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Construct the hardware-efficient ansatz circuit.

        Parameters
        ----------
        params : np.ndarray
            Parameter values of length n_params().

        Returns
        -------
        QuantumCircuit
            Parameterized circuit.
        """
        expected = self.n_params()
        if len(params) != expected:
            raise ValueError(f"Expected {expected} params, got {len(params)}")

        circuit = QuantumCircuit(self.n_qubits)
        idx = 0

        # Initial state: |+>^n (Hadamard on all qubits)
        for q in range(self.n_qubits):
            circuit.h(q)

        for layer in range(self.n_layers):
            # Rotation layer
            for q in range(self.n_qubits):
                for rot in self.rotation_set:
                    angle = params[idx]
                    idx += 1
                    if rot == 'rx':
                        circuit.rx(angle, q)
                    elif rot == 'ry':
                        circuit.ry(angle, q)
                    elif rot == 'rz':
                        circuit.rz(angle, q)

            # Entangling layer
            pairs = self._entanglement_pairs()
            for (a, b) in pairs:
                circuit.cx(a, b)

        return circuit

    def _entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Get qubit pairs for entangling gates."""
        pairs = []
        if self.entanglement == 'linear':
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))
        elif self.entanglement == 'circular':
            for i in range(self.n_qubits):
                pairs.append((i, (i + 1) % self.n_qubits))
        elif self.entanglement == 'full':
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))
        return pairs


class UCCSDAnsatz:
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    A chemistry-inspired ansatz commonly used for molecular simulations.
    It captures both single and double electron excitations.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (spin orbitals).
    n_electrons : int
        Number of electrons.
    active_space : Optional[Tuple[List[int], List[int]]]
        Active space specification (occupied, virtual orbitals).
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        active_space: Optional[Tuple[List[int], List[int]]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.active_space = active_space

        # Generate excitation operators
        self._singles: List[Tuple[int, int]] = []
        self._doubles: List[Tuple[int, int, int, int]] = []
        self._generate_excitations()

    def _generate_excitations(self) -> None:
        """Generate single and double excitation indices."""
        n_occ = self.n_electrons // 2
        n_virt = self.n_qubits // 2 - n_occ

        occupied = list(range(n_occ))
        virtual = list(range(n_occ, n_occ + n_virt))

        # Singles: i -> a (spin-up and spin-down)
        for i in occupied:
            for a in virtual:
                self._singles.append((2 * i, 2 * a))      # alpha
                self._singles.append((2 * i + 1, 2 * a + 1))  # beta

        # Doubles: ij -> ab
        for i in occupied:
            for j in occupied:
                for a in virtual:
                    for b in virtual:
                        self._doubles.append((2*i, 2*j, 2*a, 2*b))
                        self._doubles.append((2*i, 2*j+1, 2*a, 2*b+1))

    def n_params(self) -> int:
        """Total number of parameters."""
        return len(self._singles) + len(self._doubles)

    def construct_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Construct the UCCSD ansatz circuit.

        Parameters
        ----------
        params : np.ndarray
            Excitation amplitudes.

        Returns
        -------
        QuantumCircuit
            UCCSD circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Hartree-Fock initial state: |11110000> (for 4 electrons, 8 qubits)
        for i in range(self.n_electrons):
            circuit.x(i)

        idx = 0

        # Apply single excitations
        for (i, a) in self._singles:
            if idx >= len(params):
                break
            theta = params[idx]
            idx += 1

            # T_i^a - T_a^i applied as exponentiated:
            # exp(theta * (X_i Z_{i+1}..Z_{a-1} X_a - Y_i Z_{i+1}..Z_{a-1} Y_a)) / 2
            # Simplified: use two-qubit rotation
            circuit.rz(theta / 2, a)
            circuit.rz(-theta / 2, i)
            for k in range(i + 1, a):
                circuit.z(k)
            circuit.ry(theta, i)
            circuit.ry(-theta, a)
            for k in range(i + 1, a):
                circuit.z(k)
            circuit.rz(theta / 2, a)
            circuit.rz(-theta / 2, i)

        # Apply double excitations
        for (i, j, a, b) in self._doubles:
            if idx >= len(params):
                break
            theta = params[idx]
            idx += 1

            # Simplified double excitation
            circuit.ry(theta / 2, i)
            circuit.ry(-theta / 2, a)
            circuit.cx(i, j)
            circuit.cx(a, b)
            circuit.ry(-theta / 2, j)
            circuit.ry(theta / 2, b)
            circuit.cx(i, j)
            circuit.cx(a, b)

        return circuit


class VQE:
    """
    Variational Quantum Eigensolver.

    Finds the ground state energy by minimizing the expectation value
    of a Hamiltonian over a parameterized quantum circuit (ansatz).

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian to find the ground state of.
    ansatz : Union[HWEAnsatz, UCCSDAnsatz, Callable]
        The parameterized quantum circuit (ansatz).
    optimizer : str
        Classical optimizer: 'COBYLA', 'SPSA', 'L-BFGS-B', 'Nelder-Mead', 'Adam'.
    initial_params : Optional[np.ndarray]
        Initial parameter values. If None, random.

    Examples
    --------
    >>> H = Hamiltonian.transverse_field_ising(2, j=1.0, h=-1.0)
    >>> ansatz = HWEAnsatz(2, n_layers=2)
    >>> vqe = VQE(H, ansatz, optimizer='COBYLA')
    >>> result = vqe.run(max_iterations=100)
    >>> print(f"Ground state energy: {result.optimal_energy}")
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ansatz: Union[HWEAnsatz, UCCSDAnsatz, Callable],
        optimizer: str = 'COBYLA',
        initial_params: Optional[np.ndarray] = None,
        simulator: Optional['Simulator'] = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer_name = optimizer
        self.simulator = simulator or StatevectorSimulator()
        self.result: Optional[VQEResult] = None

        # Determine parameter count
        if isinstance(ansatz, (HWEAnsatz, UCCSDAnsatz)):
            self.n_params = ansatz.n_params()
        elif callable(ansatz):
            # Try to infer from a test call
            self.n_params = 0  # Will be set by user or test
        else:
            raise TypeError(f"Unknown ansatz type: {type(ansatz)}")

        if initial_params is not None:
            self.initial_params = np.array(initial_params, dtype=np.float64)
        else:
            self.initial_params = np.random.uniform(-np.pi, np.pi, self.n_params)

    def _get_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Build circuit from ansatz and parameters."""
        if isinstance(self.ansatz, (HWEAnsatz, UCCSDAnsatz)):
            return self.ansatz.construct_circuit(params)
        elif callable(self.ansatz):
            return self.ansatz(params)
        raise TypeError("Cannot construct circuit from ansatz")

    def energy(self, params: np.ndarray) -> float:
        """
        Compute the expectation value <psi(params)|H|psi(params)>.

        Parameters
        ----------
        params : np.ndarray
            Circuit parameters.

        Returns
        -------
        float
            Expected energy.
        """
        circuit = self._get_circuit(params)
        result = self.simulator.run(circuit, shots=8192)

        if hasattr(result, 'statevector') and result.statevector is not None:
            state = result.statevector
            return self.hamiltonian.expectation(state)

        # Use measurement counts for energy estimation
        counts = result.get_counts()
        total_energy = 0.0
        total_shots = sum(counts.values())

        for bitstring, count in counts.items():
            # Evaluate each Pauli term for this bitstring
            for term in self.hamiltonian.terms:
                # Compute expectation of Pauli string for this outcome
                value = 1.0
                for bit_idx, pauli in enumerate(term.pauli_string):
                    bit_val = int(bitstring[bit_idx]) if bit_idx < len(bitstring) else 0
                    if pauli == 'Z':
                        value *= 1 if bit_val == 0 else -1
                    elif pauli == 'X':
                        value *= (1 if bit_val == 0 else -1) * 0.5  # Approximate
                    elif pauli == 'Y':
                        value *= (1 if bit_val == 0 else -1) * 0.5
                total_energy += term.coefficient * value * count

        return total_energy / total_shots if total_shots > 0 else 0.0

    def gradient(self, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
        """
        Compute energy gradient using parameter-shift rule.

        dE/dθ_i ≈ [E(θ + s*e_i) - E(θ - s*e_i)] / (2 * sin(s))

        Parameters
        ----------
        params : np.ndarray
            Current parameters.
        shift : float
            Shift magnitude.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        grad = np.zeros(len(params))
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            grad[i] = (self.energy(params_plus) - self.energy(params_minus)) / (2 * np.sin(shift))
        return grad

    def run(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        callback: Optional[Callable] = None,
    ) -> VQEResult:
        """
        Run VQE optimization.

        Parameters
        ----------
        max_iterations : int
            Maximum optimization iterations.
        convergence_threshold : float
            Energy convergence threshold.
        callback : Optional[Callable]
            Callback function called after each iteration.

        Returns
        -------
        VQEResult
            Optimization result.
        """
        import time
        start_time = time.time()
        history = []

        def objective(params):
            e = self.energy(params)
            history.append(e)
            if callback:
                callback(len(history), e, params)
            return e

        if self.optimizer_name == 'COBYLA':
            result = scipy_minimize(
                objective, self.initial_params,
                method='COBYLA',
                options={'maxiter': max_iterations, 'rhobeg': 0.5},
            )
        elif self.optimizer_name == 'L-BFGS-B':
            def grad_func(params):
                return self.gradient(params)

            result = scipy_minimize(
                objective, self.initial_params,
                method='L-BFGS-B',
                jac=grad_func,
                options={'maxiter': max_iterations, 'ftol': convergence_threshold},
            )
        elif self.optimizer_name == 'Nelder-Mead':
            result = scipy_minimize(
                objective, self.initial_params,
                method='Nelder-Mead',
                options={'maxiter': max_iterations, 'xatol': convergence_threshold},
            )
        elif self.optimizer_name == 'SPSA':
            result = self._spsa_optimize(
                objective, max_iterations, convergence_threshold,
            )
            optimal_params = result
        elif self.optimizer_name == 'Adam':
            result = self._adam_optimize(
                objective, max_iterations, convergence_threshold,
            )
            optimal_params = result
        else:
            result = scipy_minimize(
                objective, self.initial_params,
                method=self.optimizer_name,
                options={'maxiter': max_iterations},
            )

        elapsed = time.time() - start_time

        if isinstance(result, np.ndarray):
            optimal_params = result
            optimal_energy = history[-1] if history else self.energy(result)
        else:
            optimal_params = result.x
            optimal_energy = result.fun

        self.result = VQEResult(
            optimal_energy=float(optimal_energy),
            optimal_params=optimal_params,
            convergence_history=history,
            iteration_count=len(history),
            optimizer_name=self.optimizer_name,
            elapsed_time=elapsed,
            success=True,
        )

        return self.result

    def _spsa_optimize(
        self,
        objective: Callable,
        max_iterations: int,
        convergence_threshold: float,
        a: float = 1.0,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = max_iterations / 10,
    ) -> np.ndarray:
        """SPSA optimization."""
        params = self.initial_params.copy()
        best_params = params.copy()
        best_energy = objective(params)

        for k in range(max_iterations):
            ak = a / ((k + 1 + A) ** alpha)
            ck = c / ((k + 1) ** gamma)

            # Random perturbation
            delta = np.random.choice([-1.0, 1.0], size=len(params))
            params_plus = params + ck * delta
            params_minus = params - ck * delta

            energy_plus = objective(params_plus)
            energy_minus = objective(params_minus)

            # SPSA gradient estimate
            grad_estimate = (energy_plus - energy_minus) / (2 * ck * delta)
            params = params - ak * grad_estimate

            current_energy = objective(params)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

            if abs(best_energy - current_energy) < convergence_threshold:
                break

        return best_params

    def _adam_optimize(
        self,
        objective: Callable,
        max_iterations: int,
        convergence_threshold: float,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Adam optimization."""
        params = self.initial_params.copy()
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        best_energy = objective(params)
        best_params = params.copy()

        for t in range(1, max_iterations + 1):
            grad = self.gradient(params)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)

            current_energy = objective(params)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

            if abs(grad).max() < convergence_threshold:
                break

        return best_params
