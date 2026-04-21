"""
Quantum Approximate Optimization Algorithm (QAOA)
==================================================

Implements the QAOA for solving combinatorial optimization problems.

QAOA alternates between a cost Hamiltonian and a mixer Hamiltonian,
with parameters optimized classically to minimize/maximize the cost
function expectation value.

References:
    - Farhi, E., Goldstone, J., Gutmann, S. (2014). A Quantum Approximate
      Optimization Algorithm.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy.optimize import minimize as scipy_minimize

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, RXGate, RYGate, RZGate,
        CNOTGate, CZGate, RXXGate, RYYGate, RZZGate,
        Measurement, UnitaryGate, PhaseGate,
    )
    from quantumflow.core.state import Statevector
    from quantumflow.simulation.simulator import StatevectorSimulator
except ImportError:
    pass


@dataclass
class QAOAResult:
    """Results from QAOA optimization."""
    optimal_cost: float
    optimal_params: np.ndarray
    best_bitstring: str
    cost_history: List[float] = field(default_factory=list)
    iteration_count: int = 0
    approximation_ratio: float = 0.0
    optimizer_name: str = ""
    success: bool = True


class CostHamiltonian:
    """
    Cost Hamiltonian for QAOA.

    Encodes the optimization problem as a Hamiltonian whose ground state
    corresponds to the optimal solution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    terms : List[Tuple[float, str]]
        List of (coefficient, pauli_string) terms.
    """

    def __init__(
        self,
        n_qubits: int,
        terms: Optional[List[Tuple[float, str]]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.terms: List[Tuple[float, str]] = terms or []

    def add_term(self, coefficient: float, pauli_string: str) -> None:
        self.terms.append((coefficient, pauli_string))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'CostHamiltonian':
        """Create from Hermitian matrix."""
        from quantumflow.algorithms.vqe import Hamiltonian
        h = Hamiltonian.from_matrix(matrix)
        ch = cls(h.n_qubits)
        for term in h.terms:
            ch.add_term(term.coefficient, term.pauli_string)
        return ch

    def matrix(self) -> np.ndarray:
        """Compute full cost Hamiltonian matrix."""
        pauli_matrices = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        result = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=np.complex128)
        for coeff, pauli in self.terms:
            mat = pauli_matrices[pauli[0]]
            for p in pauli[1:]:
                mat = np.kron(mat, pauli_matrices[p])
            result += coeff * mat
        return result

    def evaluate(self, bitstring: str) -> float:
        """Evaluate cost function for a given bitstring."""
        cost = 0.0
        for coeff, pauli in self.terms:
            val = 1.0
            for i, p in enumerate(pauli):
                if i < len(bitstring):
                    bit = int(bitstring[i])
                else:
                    bit = 0
                if p == 'Z':
                    val *= 1 if bit == 0 else -1
                elif p == 'X':
                    val *= 1
                elif p == 'Y':
                    val *= 1
            cost += coeff * val
        return cost

    def expectation(self, counts: Dict[str, int]) -> float:
        """Compute expected cost from measurement counts."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        expected = 0.0
        for bitstring, count in counts.items():
            expected += self.evaluate(bitstring) * count
        return expected / total


class MixerHamiltonian:
    """
    Mixer Hamiltonian for QAOA.

    The mixer drives transitions between states to explore the solution space.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    mixer_type : str
        Type of mixer: 'x' (standard), 'y', 'xy', 'ring'.
    """

    def __init__(self, n_qubits: int, mixer_type: str = 'x') -> None:
        self.n_qubits = n_qubits
        self.mixer_type = mixer_type

    def apply(self, circuit: QuantumCircuit, beta: float, qubits: Optional[List[int]] = None) -> None:
        """Apply the mixer unitary exp(-i*beta*H_mixer) to the circuit."""
        if qubits is None:
            qubits = list(range(self.n_qubits))

        if self.mixer_type == 'x':
            # RX rotations: exp(-i*beta*X) for each qubit
            for q in qubits:
                circuit.rx(2 * beta, q)
        elif self.mixer_type == 'y':
            for q in qubits:
                circuit.ry(2 * beta, q)
        elif self.mixer_type == 'xy':
            # XY mixer: exp(-i*beta*(X_i X_{i+1} + Y_i Y_{i+1}))
            for i in range(len(qubits) - 1):
                circuit.rxx(2 * beta, qubits[i], qubits[i + 1])
                circuit.ryy(2 * beta, qubits[i], qubits[i + 1])
        elif self.mixer_type == 'ring':
            for i in range(len(qubits)):
                j = (i + 1) % len(qubits)
                circuit.rxx(2 * beta, qubits[i], qubits[j])


class QAOA:
    """
    Quantum Approximate Optimization Algorithm.

    QAOA uses alternating layers of cost and mixer unitaries to
    approximate solutions to combinatorial optimization problems.

    Parameters
    ----------
    cost_hamiltonian : CostHamiltonian
        The problem-specific cost Hamiltonian.
    p : int
        Number of QAOA layers (depth).
    mixer : str
        Mixer type: 'x', 'y', 'xy', 'ring'.
    initial_params : Optional[np.ndarray]
        Initial parameters [gamma_0, beta_0, gamma_1, beta_1, ...].

    Examples
    --------
    >>> cost = CostHamiltonian(3, [(1.0, "ZIZ"), (1.0, "IZI"), (-1.0, "ZZI")])
    >>> qaoa = QAOA(cost, p=3)
    >>> result = qaoa.run()
    >>> print(f"Optimal cost: {result.optimal_cost}")
    """

    def __init__(
        self,
        cost_hamiltonian: CostHamiltonian,
        p: int = 1,
        mixer: str = 'x',
        initial_params: Optional[np.ndarray] = None,
    ) -> None:
        self.cost_hamiltonian = cost_hamiltonian
        self.p = p
        self.n_qubits = cost_hamiltonian.n_qubits
        self.mixer = MixerHamiltonian(self.n_qubits, mixer)
        self.result: Optional[QAOAResult] = None

        if initial_params is not None:
            self.initial_params = np.array(initial_params, dtype=np.float64)
        else:
            self.initial_params = np.random.uniform(0, 2 * np.pi, 2 * p)

    def construct_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """
        Construct the QAOA circuit.

        Parameters
        ----------
        params : np.ndarray
            Parameters [gamma_0, beta_0, gamma_1, beta_1, ...] of length 2*p.

        Returns
        -------
        QuantumCircuit
            QAOA circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Step 1: Initial state (equal superposition)
        for q in range(self.n_qubits):
            circuit.h(q)

        # Step 2: Apply p layers of cost and mixer
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]

            # Apply cost unitary: exp(-i*gamma*H_cost)
            self._apply_cost_unitary(circuit, gamma)

            # Apply mixer unitary: exp(-i*beta*H_mixer)
            self.mixer.apply(circuit, beta)

        # Step 3: Measurement
        for q in range(self.n_qubits):
            circuit.append(Measurement(), [q])

        return circuit

    def _apply_cost_unitary(self, circuit: QuantumCircuit, gamma: float) -> None:
        """Apply exp(-i*gamma*H_cost) to the circuit."""
        for coeff, pauli in self.cost_hamiltonian.terms:
            angle = -2 * coeff * gamma
            for i, p in enumerate(pauli):
                if p == 'Z':
                    circuit.rz(angle, i)
                elif p == 'X':
                    circuit.rx(angle, i)
                elif p == 'Y':
                    circuit.ry(angle, i)
                # Two-qubit terms (simplified)
                if i + 1 < len(pauli):
                    if pauli[i] == 'Z' and pauli[i+1] == 'Z':
                        circuit.rzz(angle, i, i+1)

    def cost_function(self, params: np.ndarray, simulator: Optional['StatevectorSimulator'] = None) -> float:
        """Compute cost function value for given parameters."""
        if simulator is None:
            simulator = StatevectorSimulator()

        circuit = self.construct_circuit(params)
        result = simulator.run(circuit, shots=4096)
        counts = result.get_counts()
        return self.cost_hamiltonian.expectation(counts)

    def run(
        self,
        optimizer: str = 'COBYLA',
        max_iterations: int = 100,
        simulator: Optional['StatevectorSimulator'] = None,
        callback: Optional[Callable] = None,
    ) -> QAOAResult:
        """
        Run QAOA optimization.

        Parameters
        ----------
        optimizer : str
            Classical optimizer name.
        max_iterations : int
            Maximum iterations.
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        callback : Optional[Callable]
            Callback function.

        Returns
        -------
        QAOAResult
            Optimization results.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        history = []

        def objective(params):
            cost = self.cost_function(params, simulator)
            history.append(cost)
            if callback:
                callback(len(history), cost, params)
            return cost

        result = scipy_minimize(
            objective, self.initial_params,
            method=optimizer,
            options={'maxiter': max_iterations},
        )

        # Get final measurement results
        circuit = self.construct_circuit(result.x)
        final_result = simulator.run(circuit, shots=8192)
        counts = final_result.get_counts()

        best_bitstring = max(counts, key=counts.get)

        # Compute approximation ratio
        costs = [self.cost_hamiltonian.evaluate(bs) for bs in counts]
        optimal_classical = min(costs) if costs else 0
        approx_ratio = result.fun / optimal_classical if optimal_classical != 0 else 0

        self.result = QAOAResult(
            optimal_cost=result.fun,
            optimal_params=result.x,
            best_bitstring=best_bitstring,
            cost_history=history,
            iteration_count=len(history),
            approximation_ratio=approx_ratio,
            optimizer_name=optimizer,
        )

        return self.result


class MaxCutQAOA:
    """
    QAOA specifically for the MaxCut problem.

    Given a graph G = (V, E), MaxCut finds a partition of vertices
    into two sets S and V\\S that maximizes the number of edges
    crossing the cut.

    The cost Hamiltonian is:
        H_C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of edges as (node_i, node_j) pairs.
    n_nodes : int
        Number of nodes in the graph.
    p : int
        QAOA depth.

    Examples
    --------
    >>> edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
    >>> maxcut = MaxCutQAOA(edges, n_nodes=4, p=3)
    >>> result = maxcut.solve()
    >>> print(f"Cut size: {result.optimal_cost}")
    >>> print(f"Partition: {result.best_bitstring}")
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        n_nodes: int,
        p: int = 1,
    ) -> None:
        self.edges = edges
        self.n_nodes = n_nodes
        self.p = p

        # Build cost Hamiltonian for MaxCut
        terms = []
        for (i, j) in edges:
            # MaxCut cost: (1 - Z_i Z_j) / 2
            terms.append((0.5, 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (n_nodes - j - 1)))
            terms.append((-0.5, 'I' * n_nodes))

        # Simplify: constant terms
        cost = CostHamiltonian(n_nodes)
        for (i, j) in edges:
            zz_str = ['I'] * n_nodes
            zz_str[i] = 'Z'
            zz_str[j] = 'Z'
            cost.add_term(-0.5, ''.join(zz_str))

        self.cost_hamiltonian = cost
        self.qaoa = QAOA(cost, p=p)

    def solve(
        self,
        optimizer: str = 'COBYLA',
        max_iterations: int = 100,
        shots: int = 4096,
    ) -> QAOAResult:
        """
        Solve the MaxCut problem using QAOA.

        Returns
        -------
        QAOAResult
            Result with optimal cut and partition.
        """
        return self.qaoa.run(optimizer=optimizer, max_iterations=max_iterations)

    def get_cut(self, bitstring: str) -> Tuple[List[int], List[int]]:
        """
        Extract the cut from a solution bitstring.

        Parameters
        ----------
        bitstring : str
            Solution bitstring (0s and 1s).

        Returns
        -------
        Tuple[List[int], List[int]]
            Two sets of the partition.
        """
        set_0 = [i for i, b in enumerate(bitstring) if b == '0']
        set_1 = [i for i, b in enumerate(bitstring) if b == '1']
        return set_0, set_1

    def cut_value(self, bitstring: str) -> int:
        """Compute the number of edges in the cut."""
        count = 0
        for (i, j) in self.edges:
            if bitstring[i] != bitstring[j]:
                count += 1
        return count


class MISQAOA:
    """
    QAOA for Maximum Independent Set (MIS) problem.

    An independent set is a set of vertices where no two are adjacent.
    MIS finds the largest such set.

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        Graph edges.
    n_nodes : int
        Number of nodes.
    p : int
        QAOA depth.
    """

    def __init__(self, edges: List[Tuple[int, int]], n_nodes: int, p: int = 1) -> None:
        self.edges = edges
        self.n_nodes = n_nodes
        self.p = p

        # MIS cost: maximize sum x_i - penalty * sum_{adjacent i,j} x_i * x_j
        # In QAOA, we minimize -|S| + penalty * |adjacent pairs|
        terms = []
        penalty = 2.0  # Penalty weight

        # Maximize |S|: minimize -sum Z_i (mapping 1->-1, 0->+1)
        # Using Z basis: x_i = (1 - Z_i) / 2
        # Sum x_i = (n - sum Z_i) / 2
        for i in range(n_nodes):
            pauli = ['I'] * n_nodes
            pauli[i] = 'Z'
            terms.append((0.5, ''.join(pauli)))

        # Penalty for adjacent pairs
        for (i, j) in edges:
            pauli = ['I'] * n_nodes
            pauli[i] = 'Z'
            pauli[j] = 'Z'
            terms.append((-penalty / 4, ''.join(pauli)))

        cost = CostHamiltonian(n_nodes, terms)
        self.qaoa = QAOA(cost, p=p)

    def solve(self, optimizer: str = 'COBYLA', max_iterations: int = 100) -> QAOAResult:
        """Solve the MIS problem."""
        return self.qaoa.run(optimizer=optimizer, max_iterations=max_iterations)

    def get_independent_set(self, bitstring: str) -> List[int]:
        """Extract independent set from bitstring (1 = in set)."""
        return [i for i, b in enumerate(bitstring) if b == '1']


class TSPQAOA:
    """
    QAOA for the Traveling Salesman Problem (TSP).

    Encodes TSP as a QUBO problem and solves with QAOA.

    Parameters
    ----------
    distance_matrix : np.ndarray
        N x N distance matrix.
    p : int
        QAOA depth.
    penalty_weight : float
        Weight for constraint violations.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        p: int = 1,
        penalty_weight: float = 10.0,
    ) -> None:
        self.distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        self.n_cities = distance_matrix.shape[0]
        self.p = p
        self.penalty_weight = penalty_weight

        # Variables: x_{i,t} = 1 if city i is visited at time t
        # n_qubits = n_cities^2
        self.n_qubits = self.n_cities ** 2

        terms = self._build_cost_terms()
        self.cost_hamiltonian = CostHamiltonian(self.n_qubits, terms)
        self.qaoa = QAOA(self.cost_hamiltonian, p=p)

    def _build_cost_terms(self) -> List[Tuple[float, str]]:
        """Build QUBO cost terms for TSP."""
        n = self.n_cities
        terms = []
        pw = self.penalty_weight

        # Objective: minimize total distance
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for t in range(n):
                    t_next = (t + 1) % n
                    # x_{i,t} * x_{j,t+1}
                    idx_i_t = i * n + t
                    idx_j_t = j * n + t_next

                    pauli = ['I'] * self.n_qubits
                    pauli[idx_i_t] = 'Z'
                    pauli[idx_j_t] = 'Z'
                    # Z mapping: 1 -> -1, 0 -> +1
                    # x_i * x_j = ((1-Z_i)/2) * ((1-Z_j)/2)
                    # = (1 - Z_i - Z_j + Z_i Z_j) / 4
                    terms.append((-self.distance_matrix[i, j] / 4, ''.join(pauli)))

        # Constraint: each city visited exactly once
        for i in range(n):
            for t1 in range(n):
                for t2 in range(t1 + 1, n):
                    idx1 = i * n + t1
                    idx2 = i * n + t2
                    pauli = ['I'] * self.n_qubits
                    pauli[idx1] = 'Z'
                    pauli[idx2] = 'Z'
                    terms.append((pw / 4, ''.join(pauli)))

        # Constraint: each time slot has exactly one city
        for t in range(n):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    idx1 = i1 * n + t
                    idx2 = i2 * n + t
                    pauli = ['I'] * self.n_qubits
                    pauli[idx1] = 'Z'
                    pauli[idx2] = 'Z'
                    terms.append((pw / 4, ''.join(pauli)))

        return terms

    def solve(self, optimizer: str = 'COBYLA', max_iterations: int = 100) -> QAOAResult:
        """Solve the TSP problem."""
        return self.qaoa.run(optimizer=optimizer, max_iterations=max_iterations)

    def decode_tour(self, bitstring: str) -> List[int]:
        """
        Decode the solution bitstring into a tour.

        Parameters
        ----------
        bitstring : str
            Solution bitstring.

        Returns
        -------
        List[int]
            Ordered list of cities in the tour.
        """
        n = self.n_cities
        tour = []
        for t in range(n):
            for i in range(n):
                idx = i * n + t
                if idx < len(bitstring) and bitstring[idx] == '1':
                    tour.append(i)
                    break
        return tour
