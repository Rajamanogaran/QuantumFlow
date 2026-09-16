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
from typing import Optional, List, Dict, Any

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import (
    ControlledGate, Measurement, UnitaryGate,
)
from quantumflow.core.state import Statevector
from quantumflow.simulation.simulator import StatevectorSimulator


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
        self.unitary.shape[0]
        expected = 2 ** n_state_qubits
        if self.unitary.shape != (expected, expected):
            raise ValueError(
                f"Unitary shape {self.unitary.shape} doesn't match "
                f"n_state_qubits={n_state_qubits} (expected {expected}x{expected})"
            )
        identity = np.eye(expected, dtype=np.complex128)
        if not np.allclose(self.unitary @ self.unitary.conj().T, identity, atol=1e-6):
            raise ValueError("Input matrix is not unitary")

    def _controlled_power(self, power: int, control_qubit: int) -> np.ndarray:
        """
        Compute the controlled-U^{power} matrix acting on the full register.

        The gate acts on ``eval_qubits + state_qubits`` (in that order) and
        is controlled on the single evaluation qubit at position
        ``control_qubit`` within the evaluation register.

        Parameters
        ----------
        power : int
            Exponent of U.
        control_qubit : int
            Index of the controlling evaluation qubit (0-based, MSB-first
            within the evaluation register).

        Returns
        -------
        numpy.ndarray
            A (2^(n_e+n_s) x 2^(n_e+n_s)) unitary matrix.
        """
        n_e = self.n_evaluation_qubits
        n_s = self.n_state_qubits
        U_power = np.linalg.matrix_power(self.unitary, power)

        dim = 2 ** (n_e + n_s)
        eval_dim = 2 ** n_e
        2 ** n_s

        # Simulator convention: qubit 0 is the most significant bit of the
        # basis-state index. The full index is laid out as
        #   [e_0 ... e_{n_e-1} | s_0 ... s_{n_s-1}]
        # so I = m * 2^n_s + s where m is the eval-register value and s the
        # state-register value. The gate acts on the low state bits with a
        # phase of U^power whenever eval qubit k is 1.
        n_state = 2 ** n_s
        ctrl = np.eye(dim, dtype=np.complex128)
        for m in range(eval_dim):
            if (m >> (n_e - 1 - control_qubit)) & 1:
                rows = m * n_state + np.arange(n_state)
                ctrl[np.ix_(rows, rows)] = U_power
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

        # Step 3: Controlled-U^{2^k} operations, each controlled on its own
        # evaluation qubit k
        for k in range(n_e):
            power = 2 ** k
            ctrl_gate = self._controlled_power(power, control_qubit=k)
            all_qubits = eval_qubits + state_qubits
            circuit.append(UnitaryGate(ctrl_gate, name=f"CU^{power}"), all_qubits)

        # Step 4: Inverse QFT on evaluation register. The phase winding
        # from the controlled-U^{2^k} layer is indexed in bit-reversed
        # order (qubit k carries weight 2^k while qubit 0 is the most
        # significant bit), so the swap layer of the QFT must NOT be
        # applied: the swap-free IQFT exactly inverts that state.
        from quantumflow.algorithms.qft import InverseQFT
        iqft = InverseQFT(n_e, do_swaps=False)
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

        # Convert to phase. Qubit 0 is the most significant bit of the
        # measured bitstring, and the IQFT (with swaps) yields the phase
        # directly as a binary fraction in that reading order — no
        # bit-reversal needed.
        phase_bits = best_bitstring[:self.n_evaluation_qubits]
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



def _append_controlled_unitary(
    circuit: QuantumCircuit,
    unitary: np.ndarray,
    control: int,
    state_qubits: List[int],
) -> None:
    """Append an exact controlled-U on ``[control] + state_qubits``.

    Uses the core :class:`ControlledGate`, whose matrix is built as the
    block-diagonal ``diag(I, U)`` with the control as the most significant
    qubit. An earlier implementation tried to decompose arbitrary U via
    per-element ``rz-cx`` gadgets, which only works for diagonal U and
    produced wrong results for anything else.
    """
    gate = ControlledGate(UnitaryGate(np.asarray(unitary, dtype=np.complex128)),
                          n_controls=1)
    circuit.append(gate, [control] + list(state_qubits))


def _prepare_eigenstate(circuit: QuantumCircuit, eigenstate, state_qubits: List[int]) -> None:
    """Prepare the QPE input eigenstate on *state_qubits*.

    Defaults to ``|1>`` (the last state qubit set) when no eigenstate is
    supplied. Preparing ``|0...0>`` — as an earlier version effectively
    did — zeroes out all phase information since U|0> may equal |0>.
    """
    if eigenstate is not None:
        prep = Statevector(eigenstate).to_circuit()
        circuit.compose(prep, qubits=state_qubits, inplace=True)
    else:
        circuit.x(state_qubits[-1])


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
        self._known_bits: Dict[int, int] = {}

    def construct_single_iteration(self, j: int) -> QuantumCircuit:
        """
        Construct the circuit estimating bit ``j`` (weight ``2^-j``) of the phase.

        Implements one round of the semi-classical iterative phase
        estimation scheme::

            H(ancilla) -> controlled-U^(2^(j-1)) -> RZ(feedback) -> H(ancilla)

        measuring the ancilla yields bit ``j`` with probability
        ``sin^2(pi * b_j / 2)`` once the less-significant bits ``b_i``
        (``i > j``) are fed back as a phase correction. Bits must therefore
        be estimated from the least significant bit upwards (handled by
        :meth:`run`).

        Parameters
        ----------
        j : int
            Bit index to estimate (1 = most significant bit of the phase,
            ``n_iterations`` = least significant).

        Returns
        -------
        QuantumCircuit
            Circuit for a single IPE round (includes measurement).
        """
        total = 1 + self.n_state_qubits
        circuit = QuantumCircuit(total)

        ancilla = 0
        state_qubits = list(range(1, total))

        # Prepare the eigenstate of U on the state register.
        _prepare_eigenstate(circuit, self.eigenstate, state_qubits)

        # Hadamard on ancilla
        circuit.h(ancilla)

        # Controlled-U^{2^(j-1)}
        power = 2 ** (j - 1)
        U_power = np.linalg.matrix_power(self.unitary, power)
        _append_controlled_unitary(circuit, U_power, ancilla, state_qubits)

        # Phase feedback from already-estimated less-significant bits:
        # subtract the tail digits 0.0 b_{j+1} b_{j+2} ... so that the
        # remaining phase is exactly b_j / 2 and the measurement returns
        # bit j with certainty (sin^2(pi * b_j / 2)). The correction MUST
        # act on the ancilla *after* the controlled-U and *before* the
        # final Hadamard — an RZ applied to |0> before the first H is a
        # mere global phase and has no effect.
        if self._known_bits:
            theta = -2.0 * np.pi * sum(
                b / (2.0 ** (i - j + 1)) for i, b in self._known_bits.items()
            )
            circuit.rz(float(theta), ancilla)

        # Final Hadamard and measurement
        circuit.h(ancilla)
        circuit.append(Measurement(), [ancilla])

        return circuit

    def run(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        shots_per_iteration: int = 1024,
    ) -> Dict[str, Any]:
        """
        Execute iterative phase estimation.

        Bits are estimated from the least significant bit upwards, feeding
        each result back as a phase correction for the next round.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        shots_per_iteration : int
            Shots per iteration.

        Returns
        -------
        Dict[str, Any]
            Results with 'phase', 'bit_estimates' (MSB-first),
            'confidence_history'.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        self._known_bits: Dict[int, int] = {}
        self._bit_estimates = []
        confidence_history = []

        for j in range(self.n_iterations, 0, -1):
            circuit = self.construct_single_iteration(j)
            result = simulator.run(circuit, shots=shots_per_iteration)
            counts = result.get_counts()

            # The ancilla is the first (most significant) bit of the
            # measured bitstring.
            count_1 = sum(v for bs, v in counts.items() if bs[0] == '1')
            p1 = count_1 / max(shots_per_iteration, 1)
            bit = 1 if p1 > 0.5 else 0
            confidence_history.append(max(p1, 1.0 - p1))

            self._known_bits[j] = bit

        self._bit_estimates = [self._known_bits[j] for j in range(1, self.n_iterations + 1)]
        self._phase_estimate = float(sum(
            b / (2.0 ** i) for i, b in self._known_bits.items()
        ))

        return {
            'phase': self._phase_estimate,
            'bit_estimates': self._bit_estimates,
            'confidence_history': confidence_history,
            'mean_confidence': float(np.mean(confidence_history)),
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

    Note
    ----
    Plain H-H measurements have outcome probabilities ``sin^2(pi * phase *
    2^k)``, which are invariant under ``phase -> 1 - phase``. Bayesian
    phase estimation *without* in-circuit feedback therefore cannot
    distinguish a phase from its mirror ``1 - phase`` (both hypotheses
    receive identical likelihoods for every power). For an unambiguous
    point estimate use :class:`PhaseEstimation` or
    :class:`IterativePhaseEstimation`, whose semi-classical bit feedback
    breaks the symmetry; this class is best understood as a distributional
    estimator whose reported ``std`` honestly flags such ambiguity.
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
        self.eigenstate = None

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

        # Prepare the eigenstate of U on the state register (default |1>).
        _prepare_eigenstate(circuit, self.eigenstate, state_qubits)

        circuit.h(ancilla)

        # Controlled-U^power (exact block-diagonal construction). No
        # in-circuit phase correction: the Bayesian update over the full
        # phase grid uses the raw likelihood sin^2(pi * phase * power),
        # and the accumulated history over increasing powers resolves the
        # phase just like digital phase estimation while keeping a full
        # posterior distribution (uncertainty quantification).
        U_power = np.linalg.matrix_power(self.unitary, power)
        _append_controlled_unitary(circuit, U_power, ancilla, state_qubits)

        circuit.h(ancilla)
        circuit.append(Measurement(), [ancilla])

        return circuit

    def update_posterior(self, measurement: int, power: int) -> None:
        """
        Update the probability distribution using Bayes' theorem.

        The likelihood must match the circuit, which applies a phase
        correction of ``-2*pi*mean(phase)*power`` before measurement:
        ``P(outcome=1 | phase) = sin^2(pi * (phase - mean) * power)``.

        Parameters
        ----------
        measurement : int
            Measurement outcome (0 or 1).
        power : int
            Power of U used in the measurement.
        """
        # Likelihood: P(measurement | phase) = sin^2(pi * phase * power)
        # for outcome 1 and cos^2 for outcome 0 (matching the circuit,
        # which applies no phase correction). A tiny floor keeps the
        # posterior strictly positive for numerical robustness.
        for i, phase in enumerate(self.phases):
            theta = 2 * np.pi * phase * power
            if measurement == 0:
                likelihood = (1 + np.cos(theta)) / 2
            else:
                likelihood = (1 - np.cos(theta)) / 2
            self.probability[i] *= max(likelihood, 1e-12)

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
            # Choose power: start LOW and increase. Early low-power
            # measurements have broad, unimodal likelihoods that localise
            # the posterior; high powers then sharpen it. Starting at a
            # high power aliases the likelihood into a comb and the
            # posterior never converges.
            power = 2 ** i

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

        # The posterior is generally multimodal (sin^2 likelihoods), so the
        # mean can fall between modes where no phase has support. Report
        # the maximum a posteriori phase instead.
        map_idx = int(np.argmax(self.probability))
        map_phase = float(self.phases[map_idx])
        std_phase = float(np.sqrt(np.sum(
            self.probability * (self.phases - map_phase) ** 2
        )))

        return {
            'phase': map_phase,
            'std': std_phase,
            'probability_distribution': self.probability,
            'phases': self.phases,
            'n_measurements': len(self._measurements),
            'measurements': self._measurements,
        }
