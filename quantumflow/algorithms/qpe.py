"""
Quantum Phase Estimation (QPE)
==============================

Implements standard and iterative Quantum Phase Estimation.

QPE is a fundamental quantum algorithm that estimates the phase phi
of an eigenvalue e^(2*pi*i*phi) of a unitary operator U. Given an
eigenstate |psi> of U with eigenvalue e^(2*pi*i*phi), QPE estimates
phi to n bits of precision using n+1 qubits.

References:
    - Kitaev, A.Y. (1995). Quantum measurements and the Abelian
      stabilizer problem.
    - Nielsen & Chuang, Chapter 5.2.
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, CNOTGate, ControlledGate, PhaseGate,
        Measurement, UnitaryGate,
    )
    from quantumflow.core.state import Statevector
    from quantumflow.simulation.simulator import StatevectorSimulator
except ImportError:
    pass


class PhaseEstimation:
    """
    Quantum Phase Estimation algorithm.

    Estimates the phase phi in U|psi> = e^(2*pi*i*phi)|psi>.

    The algorithm uses:
    1. Hadamard gates to create a superposition of phases
    2. Controlled-U^{2^k} operations to encode phase information
    3. Inverse QFT to extract the phase as a binary fraction

    Parameters
    ----------
    unitary : np.ndarray
        The unitary operator whose eigenvalue phase to estimate.
    n_evaluation_qubits : int
        Number of qubits for phase estimation (bits of precision).
    n_state_qubits : int
        Number of qubits in the eigenstate register.
    eigenstate : Optional[np.ndarray]
        Known eigenstate of U. If None, defaults to |0...0>.

    Examples
    --------
    >>> # Estimate phase of T gate: T|1> = e^(i*pi/4)|1>
    >>> U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    >>> qpe = PhaseEstimation(U, n_evaluation_qubits=4, n_state_qubits=1)
    >>> result = qpe.run()
    >>> phase = result['phase']  # Should be ~0.125
    """

    def __init__(
        self,
        unitary: np.ndarray,
        n_evaluation_qubits: int,
        n_state_qubits: int,
        eigenstate: Optional[np.ndarray] = None,
    ) -> None:
        self.unitary = np.asarray(unitary, dtype=np.complex128)
        self.n_evaluation_qubits = n_evaluation_qubits
        self.n_state_qubits = n_state_qubits
        self.eigenstate = eigenstate
        self.total_qubits = n_evaluation_qubits + n_state_qubits

        # Validate unitarity
        n = self.unitary.shape[0]
        expected = 2 ** n_state_qubits
        if self.unitary.shape != (expected, expected):
            raise ValueError(
                f"Unitary shape {self.unitary.shape} doesn't match "
                f"n_state_qubits={n_state_qubits} (expected {expected}x{expected})"
            )
        identity = np.eye(expected, dtype=np.complex128)
        if not np.allclose(self.unitary @ self.unitary.conj().T, identity, atol=1e-6):
            raise ValueError("Input matrix is not unitary")

    def _controlled_power(self, power: int) -> np.ndarray:
        """
        Compute controlled-U^{power} matrix.

        Returns a (2^n_eval + 2^n_state) x (2^n_eval + 2^n_state) matrix
        representing the controlled unitary.
        """
        n_e = self.n_evaluation_qubits
        n_s = self.n_state_qubits
        U_power = np.linalg.matrix_power(self.unitary, power)

        # Block-diagonal: I (for control=0) and U^power (for control=1)
        # For multiple state qubits, the control is per-qubit
        # Simplified: construct the full controlled gate matrix
        dim = 2 ** n_e * 2 ** n_s
        ctrl = np.eye(dim, dtype=np.complex128)

        # Control on the highest evaluation qubit
        eval_dim = 2 ** n_e
        state_dim = 2 ** n_s

        for i in range(eval_dim // 2):
            row_start = (i + eval_dim // 2) * state_dim
            col_start = (i + eval_dim // 2) * state_dim
            ctrl[row_start:row_start + state_dim,
                col_start:col_start + state_dim] = U_power

        return ctrl

    def construct_circuit(self) -> QuantumCircuit:
        """
        Construct the full QPE circuit.

        Circuit layout:
        - Qubits 0..n_eval-1: evaluation register (phase estimation)
        - Qubits n_eval..n_eval+n_state-1: state register

        Returns
        -------
        QuantumCircuit
            Complete QPE circuit.
        """
        n_e = self.n_evaluation_qubits
        n_s = self.n_state_qubits
        circuit = QuantumCircuit(self.total_qubits)

        eval_qubits = list(range(n_e))
        state_qubits = list(range(n_e, n_e + n_s))

        # Step 1: Initialize eigenstate
        if self.eigenstate is not None:
            state_sv = Statevector(self.eigenstate)
            prep_circuit = state_sv.to_circuit()
            # Map preparation to state qubits
            circuit.compose(prep_circuit, qubits=state_qubits, inplace=True)
        else:
            # Default: |1> for single qubit, |0...01> for multi-qubit
            circuit.x(state_qubits[0])

        # Step 2: Hadamard on evaluation register
        for q in eval_qubits:
            circuit.h(q)

        # Step 3: Controlled-U^{2^k} operations
        for k in range(n_e):
            power = 2 ** k
            ctrl_gate = self._controlled_power(power)
            all_qubits = eval_qubits + state_qubits
            circuit.append(UnitaryGate(ctrl_gate, name=f"CU^{power}"), all_qubits)

        # Step 4: Inverse QFT on evaluation register
        from quantumflow.algorithms.qft import InverseQFT
        iqft = InverseQFT(n_e, do_swaps=True)
        iqft_circuit = iqft.construct_circuit()
        circuit.compose(iqft_circuit, qubits=eval_qubits, inplace=True)

        # Step 5: Measure evaluation register
        for q in eval_qubits:
            circuit.append(Measurement(), [q])

        return circuit

    def run(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 4096,
    ) -> Dict[str, Any]:
        """
        Execute Phase Estimation.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        shots : int
            Number of measurement shots.

        Returns
        -------
        Dict[str, Any]
            Results containing:
            - 'phase': estimated phase (float in [0, 1))
            - 'phase_bits': phase as binary fraction string
            - 'counts': measurement outcome counts
            - 'probability': confidence of the result
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        circuit = self.construct_circuit()
        result = simulator.run(circuit, shots=shots)
        counts = result.get_counts()

        # Find the most likely outcome
        best_bitstring = max(counts, key=counts.get)
        best_count = counts[best_bitstring]

        # Convert to phase (reverse bit order for QPE convention)
        phase_bits = best_bitstring[:self.n_evaluation_qubits][::-1]
        phase = int(phase_bits, 2) / (2 ** self.n_evaluation_qubits)
        probability = best_count / shots

        return {
            'phase': phase,
            'phase_bits': phase_bits,
            'counts': counts,
            'probability': probability,
            'n_evaluation_qubits': self.n_evaluation_qubits,
            'precision': 1.0 / (2 ** self.n_evaluation_qubits),
        }

    @staticmethod
    def estimate_phase(
        unitary: np.ndarray,
        eigenstate: Optional[np.ndarray] = None,
        precision_bits: int = 8,
        shots: int = 4096,
        simulator: Optional['StatevectorSimulator'] = None,
    ) -> float:
        """
        Convenience method to estimate the phase of a unitary eigenvalue.

        Parameters
        ----------
        unitary : np.ndarray
            Unitary operator.
        eigenstate : Optional[np.ndarray]
            Eigenstate of U.
        precision_bits : int
            Bits of precision.
        shots : int
            Number of shots.
        simulator : Optional[StatevectorSimulator]
            Simulator.

        Returns
        -------
        float
            Estimated phase in [0, 1).
        """
        n_state = int(math.log2(unitary.shape[0]))
        qpe = PhaseEstimation(unitary, precision_bits, n_state, eigenstate)
        result = qpe.run(simulator, shots)
        return result['phase']


class IterativePhaseEstimation:
    """
    Iterative Quantum Phase Estimation.

    Uses a single ancilla qubit and repeated measurements to estimate
    the phase. This is more qubit-efficient than standard QPE but
    requires more circuit executions.

    The algorithm works by estimating one bit of the phase at a time,
    starting from the most significant bit, using Bayesian updating.

    Parameters
    ----------
    unitary : np.ndarray
        Unitary operator.
    n_state_qubits : int
        Number of qubits in the state register.
    n_iterations : int
        Number of bits to estimate (precision).
    eigenstate : Optional[np.ndarray]
        Known eigenstate.

    Examples
    --------
    >>> U = np.array([[1, 0], [0, np.exp(1j * np.pi / 3)]])
    >>> ipe = IterativePhaseEstimation(U, n_state_qubits=1, n_iterations=6)
    >>> result = ipe.run()
    >>> print(f"Phase: {result['phase']:.6f}")
    """

    def __init__(
        self,
        unitary: np.ndarray,
        n_state_qubits: int,
        n_iterations: int,
        eigenstate: Optional[np.ndarray] = None,
    ) -> None:
        self.unitary = np.asarray(unitary, dtype=np.complex128)
        self.n_state_qubits = n_state_qubits
        self.n_iterations = n_iterations
        self.eigenstate = eigenstate
        self._phase_estimate = 0.0
        self._bit_estimates: List[int] = []

    def construct_single_iteration(self, k: int) -> QuantumCircuit:
        """
        Construct the circuit for the k-th iteration.

        Parameters
        ----------
        k : int
            Iteration index (0 = most significant bit).

        Returns
        -------
        QuantumCircuit
            Circuit for single iteration.
        """
        # Total qubits: 1 ancilla + n_state + 1 phase kickback
        total = 1 + self.n_state_qubits
        circuit = QuantumCircuit(total)

        ancilla = 0
        state_qubits = list(range(1, total))

        # Apply previous phase correction
        correction_phase = -2 * np.pi * self._phase_estimate * (2 ** k)
        circuit.rz(correction_phase, ancilla)

        # Hadamard on ancilla
        circuit.h(ancilla)

        # Controlled-U^{2^k}
        power = 2 ** k
        U_power = np.linalg.matrix_power(self.unitary, power)

        # Apply controlled-U: if ancilla=|1>, apply U^power to state
        for s_q in state_qubits:
            for t_q in state_qubits:
                idx_s = s_q - 1
                idx_t = t_q - 1
                val = U_power[idx_s, idx_t]
                if abs(val) > 1e-12 and abs(val - 1.0) > 1e-12:
                    # Controlled phase rotation
                    phase = np.angle(val)
                    circuit.rz(phase, t_q)
                    circuit.cx(ancilla, t_q)
                    circuit.rz(-phase, t_q)
                    circuit.cx(ancilla, t_q)

        # Inverse QFT (just H for single qubit)
        circuit.h(ancilla)

        # Measure ancilla
        circuit.append(Measurement(), [ancilla])

        return circuit

    def run(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots_per_iteration: int = 1024,
    ) -> Dict[str, Any]:
        """
        Execute iterative phase estimation.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        shots_per_iteration : int
            Shots per iteration.

        Returns
        -------
        Dict[str, Any]
            Results with 'phase', 'bit_estimates', 'confidence_history'.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        self._phase_estimate = 0.0
        self._bit_estimates = []
        confidence_history = []

        for k in range(self.n_iterations):
            circuit = self.construct_single_iteration(k)
            result = simulator.run(circuit, shots=shots_per_iteration)
            counts = result.get_counts()

            # Determine the most likely bit
            count_1 = sum(v for k_str, v in counts.items() if k_str.startswith('1'))
            count_0 = shots_per_iteration - count_1

            bit = 1 if count_1 > count_0 else 0
            confidence = max(count_0, count_1) / shots_per_iteration
            confidence_history.append(confidence)

            self._bit_estimates.append(bit)
            self._phase_estimate += bit / (2 ** (k + 1))

        return {
            'phase': self._phase_estimate,
            'bit_estimates': self._bit_estimates,
            'confidence_history': confidence_history,
            'mean_confidence': np.mean(confidence_history),
        }


class BayesianPhaseEstimation:
    """
    Bayesian approach to phase estimation.

    Maintains a probability distribution over possible phases and
    updates it with each measurement using Bayes' theorem.

    This is particularly useful for:
    - Handling noisy measurements
    - Providing uncertainty quantification
    - Adaptive measurement strategies
    """

    def __init__(
        self,
        unitary: np.ndarray,
        n_state_qubits: int,
        resolution: int = 1024,
    ) -> None:
        self.unitary = np.asarray(unitary, dtype=np.complex128)
        self.n_state_qubits = n_state_qubits
        self.resolution = resolution

        # Initialize uniform prior over phases
        self.phases = np.linspace(0, 1, resolution, endpoint=False)
        self.probability = np.ones(resolution) / resolution

        self._measurements: List[int] = []
        self._powers: List[int] = []

    def construct_measurement_circuit(self, power: int = 1) -> QuantumCircuit:
        """Construct circuit for a single measurement at given power."""
        total = 1 + self.n_state_qubits
        circuit = QuantumCircuit(total)

        ancilla = 0
        state_qubits = list(range(1, total))

        # Phase correction based on current estimate
        current_phase = np.sum(self.probability * self.phases)
        correction = -2 * np.pi * current_phase * power
        circuit.rz(correction, ancilla)
        circuit.h(ancilla)

        # Controlled-U^power
        U_power = np.linalg.matrix_power(self.unitary, power)
        for s_q in state_qubits:
            for t_q in state_qubits:
                idx_s = s_q - 1
                idx_t = t_q - 1
                val = U_power[idx_s, idx_t]
                if abs(val) > 1e-12 and abs(val - 1.0) > 1e-12:
                    phase = np.angle(val)
                    circuit.rz(phase * power, t_q)
                    circuit.cx(ancilla, t_q)
                    circuit.rz(-phase * power, t_q)
                    circuit.cx(ancilla, t_q)

        circuit.h(ancilla)
        circuit.append(Measurement(), [ancilla])

        return circuit

    def update_posterior(self, measurement: int, power: int) -> None:
        """
        Update the probability distribution using Bayes' theorem.

        Parameters
        ----------
        measurement : int
            Measurement outcome (0 or 1).
        power : int
            Power of U used in the measurement.
        """
        # Likelihood: P(measurement | phase)
        for i, phase in enumerate(self.phases):
            theta = 2 * np.pi * phase * power
            if measurement == 0:
                likelihood = (1 + np.cos(theta)) / 2
            else:
                likelihood = (1 - np.cos(theta)) / 2
            self.probability[i] *= likelihood

        # Normalize
        total = np.sum(self.probability)
        if total > 0:
            self.probability /= total

        self._measurements.append(measurement)
        self._powers.append(power)

    def run(
        self,
        max_measurements: int = 100,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 100,
    ) -> Dict[str, Any]:
        """
        Run Bayesian phase estimation.

        Parameters
        ----------
        max_measurements : int
            Maximum number of measurements.
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        shots : int
            Shots per measurement.

        Returns
        -------
        Dict[str, Any]
            Results with 'phase', 'std', 'probability_distribution'.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        for i in range(max_measurements):
            # Choose power: start high, decrease
            power = max(1, 2 ** (max_measurements - i - 1))

            circuit = self.construct_measurement_circuit(power)
            result = simulator.run(circuit, shots=shots)
            counts = result.get_counts()

            count_1 = sum(v for k_str, v in counts.items() if k_str.startswith('1'))
            measurement = 1 if count_1 > shots // 2 else 0

            self.update_posterior(measurement, power)

            # Check convergence
            phase_std = np.sqrt(np.sum(self.probability * (self.phases - np.sum(self.probability * self.phases)) ** 2))
            if phase_std < 1e-4:
                break

        mean_phase = np.sum(self.probability * self.phases)
        std_phase = np.sqrt(np.sum(self.probability * (self.phases - mean_phase) ** 2))

        return {
            'phase': mean_phase,
            'std': std_phase,
            'probability_distribution': self.probability,
            'phases': self.phases,
            'n_measurements': len(self._measurements),
            'measurements': self._measurements,
        }
