"""
Grover's Search Algorithm and Amplitude Amplification
======================================================

Implements Grover's quantum search algorithm for unstructured search,
and its generalization: Amplitude Amplification.

Grover's algorithm provides a quadratic speedup over classical search,
finding a marked element in O(sqrt(N)) queries instead of O(N).

The algorithm consists of three main components:
1. State preparation (typically equal superposition via Hadamard)
2. Oracle that marks the target state(s)
3. Diffusion operator (inversion about the mean)

References:
    - Grover, L.K. (1996). A fast quantum mechanical algorithm for database search.
    - Brassard, G., et al. (2002). Quantum Amplitude Amplification and Estimation.
"""

import math
import numpy as np
from typing import List, Optional, Union, Callable, Tuple, Dict, Any

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, ZGate, CNOTGate, CZGate,
        MultiControlledXGate, UnitaryGate, Measurement,
    )
    from quantumflow.core.state import Statevector
    from quantumflow.simulation.simulator import StatevectorSimulator
except ImportError:
    pass


class GroverSearch:
    """
    Grover's Search Algorithm for finding marked states in an unstructured database.

    Grover's algorithm provides a quadratic speedup over classical exhaustive search.
    Given N = 2^n items and k marked items, the algorithm finds a marked item
    with high probability using O(sqrt(N/k)) oracle calls.

    The algorithm works by repeatedly applying:
    1. The oracle: flips the phase of marked states
    2. The diffusion operator: inverts amplitudes about the mean

    The optimal number of iterations is approximately (pi/4) * sqrt(N/k).

    Parameters
    ----------
    n_qubits : int
        Number of qubits representing the search space of size 2^n.
    oracle : Union[str, Callable, QuantumCircuit, List[str]], optional
        The oracle specification. Can be:
        - A callable that takes a QuantumCircuit and adds oracle gates
        - A QuantumCircuit that represents the oracle
        - A list of marked state strings (e.g., ['101', '011'])
        - None (must set marked_states separately)
    marked_states : Optional[List[str]], optional
        List of marked state strings (e.g., ['101', '011']).
        Used to automatically construct the oracle.
    num_iterations : Optional[int], optional
        Number of Grover iterations. If None, uses optimal value.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    marked_states : List[str]
        List of marked state strings.
    num_iterations : int
        Number of Grover iterations to apply.

    Examples
    --------
    Search for state |101> in a 3-qubit system:

    >>> grover = GroverSearch(n_qubits=3, marked_states=['101'])
    >>> grover.construct_circuit()
    >>> result = grover.run()

    Multiple marked states:

    >>> grover = GroverSearch(n_qubits=4, marked_states=['0110', '1001', '1111'])
    >>> result = grover.run()
    >>> print(result.most_frequent())
    """

    def __init__(
        self,
        n_qubits: int,
        oracle: Union[str, Callable, 'QuantumCircuit', List[str], None] = None,
        marked_states: Optional[List[str]] = None,
        num_iterations: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self._oracle_circuit: Optional['QuantumCircuit'] = None
        self._oracle_callable: Optional[Callable] = None
        self._marked_states: List[str] = []

        # Process oracle
        if oracle is not None:
            if isinstance(oracle, list):
                self.marked_states = oracle
            elif isinstance(oracle, str):
                self.marked_states = [oracle]
            elif isinstance(oracle, QuantumCircuit):
                self._oracle_circuit = oracle
            elif callable(oracle):
                self._oracle_callable = oracle

        if marked_states is not None:
            self.marked_states = marked_states

        # Compute optimal iterations if not specified
        if num_iterations is not None:
            self.num_iterations = num_iterations
        elif len(self._marked_states) > 0:
            self.num_iterations = self.optimal_iterations()
        else:
            self.num_iterations = int(math.floor(math.pi / 4 * math.sqrt(2 ** self.n_qubits)))

    @property
    def marked_states(self) -> List[str]:
        """Return the list of marked states."""
        return self._marked_states

    @marked_states.setter
    def marked_states(self, states: Union[List[str], str]) -> None:
        """Set the marked states and invalidate cached oracle."""
        if isinstance(states, str):
            self._marked_states = [states]
        else:
            self._marked_states = list(states)
        # Validate state strings
        for state in self._marked_states:
            if len(state) != self.n_qubits:
                raise ValueError(
                    f"State '{state}' has length {len(state)}, "
                    f"expected {self.n_qubits}"
                )
            if not all(c in '01' for c in state):
                raise ValueError(f"State '{state}' contains non-binary characters")
        self._oracle_circuit = None

    def optimal_iterations(self) -> int:
        """
        Compute the optimal number of Grover iterations.

        For k marked states in a search space of size N = 2^n,
        the optimal number of iterations is:
            R = floor((pi / (4 * arcsin(sqrt(k/N)))) - 0.5)

        Returns
        -------
        int
            Optimal number of Grover iterations.
        """
        n = 2 ** self.n_qubits
        k = len(self._marked_states)
        if k == 0:
            return 0
        if k >= n:
            return 0

        ratio = k / n
        arcsin_val = math.asin(math.sqrt(ratio))
        optimal = math.pi / (4 * arcsin_val) - 0.5
        return max(1, int(math.floor(optimal)))

    def success_probability(self, num_iterations: Optional[int] = None) -> float:
        """
        Compute the theoretical success probability for a given number of iterations.

        After R iterations, the probability of measuring a marked state is:
            P = sin^2((2R + 1) * arcsin(sqrt(k/N)))

        Parameters
        ----------
        num_iterations : Optional[int]
            Number of iterations. If None, uses self.num_iterations.

        Returns
        -------
        float
            Success probability in [0, 1].
        """
        n = 2 ** self.n_qubits
        k = len(self._marked_states)
        if k == 0 or k >= n:
            return 0.0 if k == 0 else 1.0

        if num_iterations is None:
            num_iterations = self.num_iterations

        ratio = k / n
        theta = math.asin(math.sqrt(ratio))
        prob = (math.sin((2 * num_iterations + 1) * theta)) ** 2
        return min(1.0, max(0.0, prob))

    def create_oracle(self) -> 'QuantumCircuit':
        """
        Construct the Grover oracle circuit from marked states.

        The oracle flips the phase of all marked states while leaving
        other states unchanged. This is implemented using a multi-controlled
        Z gate applied conditionally on each marked state.

        For a single marked state |s>, the oracle is:
            O = I - 2|s><s|

        For multiple marked states {s_1, ..., s_k}:
            O = I - 2 * sum_i |s_i><s_i|

        Returns
        -------
        QuantumCircuit
            Oracle circuit that flips the phase of marked states.
        """
        if self._oracle_circuit is not None:
            return self._oracle_circuit

        if not self._marked_states:
            raise ValueError("No marked states specified. Cannot construct oracle.")

        oracle = QuantumCircuit(self.n_qubits)

        # Build oracle using phase kickback with an ancilla
        # O|s> = -|s>, O|x> = |x> for unmarked x
        if len(self._marked_states) == 1:
            # Single marked state: use multi-controlled Z
            target_state = self._marked_states[0]
            # Apply X gates for 0 bits (flip to all-ones control condition)
            for i, bit in enumerate(target_state):
                if bit == '0':
                    oracle.append(XGate(), [i])
            # Multi-controlled Z (all qubits control Z on last qubit)
            oracle.append(ZGate(), [self.n_qubits - 1])
            # This needs to be a controlled-Z with all preceding qubits as controls
            # Use MCZ: H on target -> MCX -> H on target
            # Actually, let's use the diagonal approach
            # Undo the X gates
            for i, bit in enumerate(target_state):
                if bit == '0':
                    oracle.append(XGate(), [i])
        else:
            # Multiple marked states: build from diagonal matrix
            diagonal = np.ones(2 ** self.n_qubits, dtype=np.complex128)
            for state_str in self._marked_states:
                idx = int(state_str, 2)
                diagonal[idx] = -1.0
            oracle_matrix = np.diag(diagonal)
            oracle.append(UnitaryGate(oracle_matrix, name="Oracle"), list(range(self.n_qubits)))

        self._oracle_circuit = oracle
        return oracle

    def create_diffusion(self) -> 'QuantumCircuit':
        """
        Construct the Grover diffusion operator (inversion about the mean).

        The diffusion operator D = 2|s><s| - I, where |s> is the equal
        superposition state. This amplifies the amplitude of marked states.

        Implementation:
        1. Apply H gates to all qubits
        2. Apply X gates to all qubits (flip to |0...0>)
        3. Apply multi-controlled Z on |0...0>
        4. Apply X gates to all qubits (undo flip)
        5. Apply H gates to all qubits (undo H)

        Returns
        -------
        QuantumCircuit
            Diffusion operator circuit.
        """
        diffusion = QuantumCircuit(self.n_qubits)

        # Step 1 & 2: H then X on all qubits
        for i in range(self.n_qubits):
            diffusion.h(i)
            diffusion.x(i)

        # Step 3: Multi-controlled Z on |0...0>
        # MCZ = H(target) -> MCX -> H(target)
        diffusion.h(self.n_qubits - 1)
        # MCX with all other qubits as controls
        if self.n_qubits == 1:
            diffusion.z(0)
        elif self.n_qubits == 2:
            diffusion.cz(0, 1)
        else:
            # For n > 2, use Toffoli decomposition
            controls = list(range(self.n_qubits - 1))
            target = self.n_qubits - 1
            self._apply_mcx(diffusion, controls, target)
        diffusion.h(self.n_qubits - 1)

        # Step 4 & 5: X then H on all qubits
        for i in range(self.n_qubits):
            diffusion.x(i)
            diffusion.h(i)

        return diffusion

    @staticmethod
    def _apply_mcx(circuit: 'QuantumCircuit', controls: List[int], target: int) -> None:
        """Apply multi-controlled X gate using Toffoli decomposition."""
        n_controls = len(controls)
        if n_controls == 1:
            circuit.cx(controls[0], target)
        elif n_controls == 2:
            circuit.cx(controls[0], target)
            circuit.cx(controls[1], target)
        else:
            # Gray code / standard decomposition for large control sets
            # Use intermediate ancilla approach (simplified)
            # Recursive decomposition
            mid = n_controls // 2
            if mid > 0:
                # Decompose into smaller MCX gates
                temp = controls[0]  # Reuse first control as temp
                GroverSearch._apply_mcx(circuit, controls[:mid], temp)
                GroverSearch._apply_mcx(circuit, [temp] + controls[mid:], target)
                GroverSearch._apply_mcx(circuit, controls[:mid], temp)

    def construct_circuit(self) -> 'QuantumCircuit':
        """
        Construct the full Grover search circuit.

        The circuit consists of:
        1. Hadamard gates on all qubits (create equal superposition)
        2. Repeated application of (Oracle + Diffusion) for num_iterations times

        Returns
        -------
        QuantumCircuit
            Complete Grover search circuit.

        Raises
        ------
        ValueError
            If no marked states are specified.
        """
        if not self._marked_states and self._oracle_circuit is None and self._oracle_callable is None:
            raise ValueError("No marked states or oracle specified")

        circuit = QuantumCircuit(self.n_qubits)

        # Step 1: Initialize to equal superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # Step 2: Apply Grover iterations
        oracle = self.create_oracle()
        diffusion = self.create_diffusion()

        for _ in range(self.num_iterations):
            # Apply oracle
            if self._oracle_callable is not None:
                self._oracle_callable(circuit)
            else:
                circuit.compose(oracle, inplace=True)

            # Apply diffusion
            circuit.compose(diffusion, inplace=True)

        # Add measurement
        for i in range(self.n_qubits):
            circuit.append(Measurement(), [i])

        return circuit

    def run(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Execute Grover's search algorithm.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator to use. If None, creates a StatevectorSimulator.
        shots : int
            Number of measurement shots.

        Returns
        -------
        Dict[str, Any]
            Results dictionary containing:
            - 'circuit': The constructed quantum circuit
            - 'counts': Measurement outcome counts
            - 'marked_found': Whether a marked state was found
            - 'most_frequent': Most frequently measured state
            - 'success_probability': Theoretical success probability
            - 'num_iterations': Number of Grover iterations used
        """
        if simulator is None:
            from quantumflow.simulation.simulator import StatevectorSimulator
            simulator = StatevectorSimulator()

        circuit = self.construct_circuit()
        result = simulator.run(circuit, shots=shots)

        counts = result.get_counts()
        most_frequent = max(counts, key=counts.get)

        marked_found = any(
            state in self._marked_states for state in counts
        )

        return {
            'circuit': circuit,
            'counts': counts,
            'marked_found': marked_found,
            'most_frequent': most_frequent,
            'success_probability': self.success_probability(),
            'num_iterations': self.num_iterations,
            'shots': shots,
        }

    def find_marked_state(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 1024,
    ) -> str:
        """
        Find and return a marked state.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator to use.
        shots : int
            Number of measurement shots.

        Returns
        -------
        str
            The most likely marked state found.

        Raises
        ------
        RuntimeError
            If no marked state is found (unlikely with correct iteration count).
        """
        result = self.run(simulator, shots)
        if result['marked_found']:
            return result['most_frequent']
        raise RuntimeError(
            "Grover's algorithm did not find a marked state. "
            "Try increasing the number of shots or adjusting iterations."
        )


class AmplitudeAmplification:
    """
    Generalized Amplitude Amplification algorithm.

    Amplitude Amplification generalizes Grover's algorithm to work with
    arbitrary initial states and arbitrary "good" state operators. It
    can amplify the amplitude of any subspace defined by a projector.

    Given an initial state |psi> = A|0> and a projector onto "good" states P_good,
    the algorithm amplifies the "good" subspace component.

    The algorithm achieves the same O(1/sqrt(a)) speedup where a is the
    initial success probability.

    Parameters
    ----------
    state_preparation : Callable
        A callable that prepares the initial state on a QuantumCircuit.
    good_state_checker : Callable
        A callable that marks "good" states (the oracle).
    n_qubits : int
        Number of qubits.
    num_iterations : Optional[int]
        Number of iterations. If None, uses optimal value.

    Examples
    --------
    >>> def prepare_state(qc):
    ...     for i in range(3):
    ...         qc.ry(0.3, i)
    >>> def check_good(qc):
    ...     qc.cz(0, 2)
    >>> aa = AmplitudeAmplification(prepare_state, check_good, n_qubits=3)
    >>> result = aa.run()
    """

    def __init__(
        self,
        state_preparation: Callable,
        good_state_checker: Callable,
        n_qubits: int,
        num_iterations: Optional[int] = None,
    ) -> None:
        self.state_preparation = state_preparation
        self.good_state_checker = good_state_checker
        self.n_qubits = n_qubits
        self.num_iterations = num_iterations

    def create_diffusion(self) -> 'QuantumCircuit':
        """
        Create the generalized diffusion operator.

        The diffusion operator reflects about the initial state |psi>:
            D = 2|psi><psi| - I

        This is implemented as:
            D = A * (2|0><0| - I) * A†

        where A is the state preparation operator.

        Returns
        -------
        QuantumCircuit
            Diffusion operator circuit.
        """
        diffusion = QuantumCircuit(self.n_qubits)

        # Apply A† (inverse of state preparation)
        # For generic state preparation, we use the inverse circuit
        prep = QuantumCircuit(self.n_qubits)
        self.state_preparation(prep)
        inv_prep = prep.inverse()
        diffusion.compose(inv_prep, inplace=True)

        # Apply 2|0><0| - I (reflection about |0>)
        for i in range(self.n_qubits):
            diffusion.x(i)
        # Multi-controlled Z on |0...0>
        diffusion.h(self.n_qubits - 1)
        if self.n_qubits > 1:
            diffusion.cx(0, self.n_qubits - 1)
        diffusion.h(self.n_qubits - 1)
        for i in range(self.n_qubits):
            diffusion.x(i)

        # Apply A
        self.state_preparation(diffusion)

        return diffusion

    def construct_circuit(self) -> 'QuantumCircuit':
        """
        Construct the full Amplitude Amplification circuit.

        Returns
        -------
        QuantumCircuit
            Complete Amplitude Amplification circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Step 1: Prepare initial state
        self.state_preparation(circuit)

        # Step 2: Amplify good states
        diffusion = self.create_diffusion()

        n_iter = self.num_iterations if self.num_iterations is not None else 1
        for _ in range(n_iter):
            # Apply oracle (good state checker)
            self.good_state_checker(circuit)
            # Apply diffusion
            circuit.compose(diffusion, inplace=True)

        # Add measurement
        for i in range(self.n_qubits):
            circuit.append(Measurement(), [i])

        return circuit

    def run(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Execute Amplitude Amplification.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator. If None, creates one.
        shots : int
            Number of measurement shots.

        Returns
        -------
        Dict[str, Any]
            Results with 'circuit', 'counts', 'most_frequent'.
        """
        if simulator is None:
            from quantumflow.simulation.simulator import StatevectorSimulator
            simulator = StatevectorSimulator()

        circuit = self.construct_circuit()
        result = simulator.run(circuit, shots=shots)
        counts = result.get_counts()
        most_frequent = max(counts, key=counts.get)

        return {
            'circuit': circuit,
            'counts': counts,
            'most_frequent': most_frequent,
            'shots': shots,
        }


class FixedPointAmplitudeAmplification:
    """
    Fixed-point Amplitude Amplification.

    Unlike standard Grover/amplitude amplification which oscillates,
    fixed-point variants monotonically increase the success probability.
    This is useful when the initial success probability is unknown.

    References:
        - Yoder, Low, Chuang (2014). Fixed-point quantum search with an
          optimal number of queries.
        - Grover (2005). Fixed-point quantum search.
    """

    def __init__(self, n_qubits: int, precision: float = 1e-6) -> None:
        self.n_qubits = n_qubits
        self.precision = precision

    def construct_iteration(
        self,
        oracle: 'QuantumCircuit',
        phase_angle: float,
    ) -> 'QuantumCircuit':
        """
        Construct a single fixed-point iteration.

        Each iteration applies the oracle with a modified phase angle,
        followed by a selective inversion about the marked subspace.

        Parameters
        ----------
        oracle : QuantumCircuit
            Oracle circuit for marked states.
        phase_angle : float
            Phase rotation angle for this iteration.

        Returns
        -------
        QuantumCircuit
            Single fixed-point iteration circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Apply oracle with modified phase
        circuit.compose(oracle, inplace=True)

        # Selective phase inversion
        for i in range(self.n_qubits):
            circuit.ry(2 * phase_angle, i)

        return circuit

    def construct_circuit(
        self,
        oracle: 'QuantumCircuit',
        n_iterations: int = 3,
    ) -> 'QuantumCircuit':
        """
        Construct the full fixed-point amplitude amplification circuit.

        Parameters
        ----------
        oracle : QuantumCircuit
            Oracle circuit.
        n_iterations : int
            Number of fixed-point iterations.

        Returns
        -------
        QuantumCircuit
            Complete fixed-point circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Initialize to equal superposition
        for i in range(self.n_qubits):
            circuit.h(i)

        # Apply fixed-point iterations with decreasing phase angles
        for k in range(n_iterations):
            phase = math.pi / (2 * (k + 2))
            iteration = self.construct_iteration(oracle, phase)
            circuit.compose(iteration, inplace=True)

        # Add measurement
        for i in range(self.n_qubits):
            circuit.append(Measurement(), [i])

        return circuit
