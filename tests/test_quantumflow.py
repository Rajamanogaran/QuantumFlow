"""
QuantumFlow Test Suite
======================
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantumflow.core.gate import (
    HGate, XGate, YGate, ZGate, CNOTGate, RXGate, RYGate, RZGate,
    UnitaryGate, CZGate, SwapGate,
)
from quantumflow.core.state import Statevector, DensityMatrix
from quantumflow.core.circuit import QuantumCircuit
from quantumflow.utils.math import (
    fidelity, purity, von_neumann_entropy, expectation_value,
    is_unitary, is_hermitian, random_unitary, kron,
)


class TestGates(unittest.TestCase):
    """Test quantum gate operations."""

    def test_hadamard(self):
        """Test H gate properties."""
        H = HGate()
        self.assertTrue(is_unitary(H.matrix))
        self.assertTrue(is_hermitian(H.matrix))

    def test_pauli_gates(self):
        """Test Pauli X, Y, Z gates."""
        for gate_cls in [XGate, YGate, ZGate]:
            gate = gate_cls()
            self.assertTrue(is_unitary(gate.matrix))

        # Test X gate flips |0> to |1>
        X = XGate()
        result = X.matrix @ np.array([1, 0])
        expected = np.array([0, 1])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_cnot(self):
        """Test CNOT gate."""
        cnot = CNOTGate()
        self.assertTrue(is_unitary(cnot.matrix))
        self.assertEqual(cnot.matrix.shape, (4, 4))

        # CNOT|00> = |00>
        state = np.array([1, 0, 0, 0])
        result = cnot.matrix @ state
        np.testing.assert_allclose(result, state, atol=1e-10)

        # CNOT|10> = |11>
        state = np.array([0, 0, 1, 0])
        expected = np.array([0, 0, 0, 1])
        result = cnot.matrix @ state
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rotation_gates(self):
        """Test rotation gates RX, RY, RZ."""
        for gate_cls in [RXGate, RYGate, RZGate]:
            gate = gate_cls(params=[np.pi / 2])
            self.assertTrue(is_unitary(gate.matrix))

    def test_unitary_gate(self):
        """Test custom unitary gate."""
        U = UnitaryGate(np.array([[0, 1], [1, 0]], dtype=np.complex128))
        np.testing.assert_allclose(U.matrix, XGate().matrix, atol=1e-10)

    def test_gate_composition(self):
        """Test that H^2 = I."""
        H = HGate()
        HH = H.matrix @ H.matrix
        np.testing.assert_allclose(HH, np.eye(2), atol=1e-10)


class TestStatevector(unittest.TestCase):
    """Test statevector operations."""

    def test_creation(self):
        """Test statevector creation."""
        sv = Statevector(np.array([1, 0]))
        np.testing.assert_allclose(sv.data, [1, 0])

    def test_probabilities(self):
        """Test measurement probabilities."""
        sv = Statevector(np.array([1, 1]) / np.sqrt(2))
        probs = sv.probabilities()
        np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-10)

    def test_from_label(self):
        """Test statevector from label."""
        sv = Statevector.from_label("00")
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_allclose(sv.data, expected, atol=1e-10)

    def test_evolve(self):
        """Test state evolution."""
        sv = Statevector(np.array([1, 0]))
        sv = sv.evolve(XGate().matrix)
        np.testing.assert_allclose(sv.data, [0, 1], atol=1e-10)

    def test_bell_state(self):
        """Test Bell state creation."""
        H = HGate()
        sv = Statevector(np.array([1, 0, 0, 0]))
        sv = sv.evolve(kron(H.matrix, np.eye(2)))
        sv = sv.evolve(CNOTGate().matrix)
        # Should be (|00> + |11>) / sqrt(2)
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sv.data, expected, atol=1e-10)

    def test_tensor(self):
        """Test tensor product of states."""
        sv1 = Statevector(np.array([1, 0]))
        sv2 = Statevector(np.array([0, 1]))
        combined = sv1.tensor(sv2)
        expected = np.array([0, 1, 0, 0])
        np.testing.assert_allclose(combined.data, expected, atol=1e-10)


class TestDensityMatrix(unittest.TestCase):
    """Test density matrix operations."""

    def test_from_statevector(self):
        """Test creation from statevector."""
        dm = DensityMatrix.from_statevector(np.array([1, 0]))
        np.testing.assert_allclose(dm.data, np.diag([1, 0]), atol=1e-10)

    def test_purity_pure(self):
        """Test purity of pure state is 1."""
        dm = DensityMatrix.from_statevector(np.array([1, 1]) / np.sqrt(2))
        self.assertAlmostEqual(dm.purity(), 1.0, places=10)

    def test_purity_mixed(self):
        """Test purity of maximally mixed state."""
        dm = DensityMatrix(np.eye(2) / 2)
        self.assertAlmostEqual(dm.purity(), 0.5, places=10)

    def test_entropy_pure(self):
        """Test entropy of pure state is 0."""
        dm = DensityMatrix.from_statevector(np.array([1, 0]))
        self.assertAlmostEqual(dm.von_neumann_entropy(), 0.0, places=10)

    def test_fidelity(self):
        """Test fidelity calculation."""
        dm1 = DensityMatrix.from_statevector(np.array([1, 0]))
        dm2 = DensityMatrix.from_statevector(np.array([1, 0]))
        self.assertAlmostEqual(dm1.fidelity(dm2), 1.0, places=10)

        dm3 = DensityMatrix.from_statevector(np.array([0, 1]))
        self.assertAlmostEqual(dm1.fidelity(dm3), 0.0, places=10)


class TestCircuit(unittest.TestCase):
    """Test quantum circuit construction."""

    def test_bell_circuit(self):
        """Test Bell state circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        U = qc.to_unitary()
        # Bell state unitary: CNOT @ (H tensor I)
        H = HGate().matrix
        CNOT = CNOTGate().matrix
        expected = CNOT @ np.kron(H, np.eye(2))
        np.testing.assert_allclose(U, expected, atol=1e-10)

    def test_ghz_circuit(self):
        """Test GHZ state circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        self.assertIsNotNone(qc)

    def test_depth(self):
        """Test circuit depth calculation."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        self.assertEqual(qc.depth(), 3)

    def test_width(self):
        """Test circuit width."""
        qc = QuantumCircuit(5)
        self.assertEqual(qc.width, 5)

    def test_inverse(self):
        """Test circuit inversion."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        inv = qc.inverse()
        U = qc.to_unitary() @ inv.to_unitary()
        np.testing.assert_allclose(U, np.eye(4), atol=1e-10)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_kron(self):
        """Test Kronecker product."""
        result = kron(np.eye(2), np.array([[0, 1], [1, 0]]))
        self.assertEqual(result.shape, (4, 4))

    def test_fidelity_same_state(self):
        """Test fidelity of identical states."""
        f = fidelity(np.array([1, 0]), np.array([1, 0]))
        self.assertAlmostEqual(f, 1.0, places=10)

    def test_fidelity_orthogonal(self):
        """Test fidelity of orthogonal states."""
        f = fidelity(np.array([1, 0]), np.array([0, 1]))
        self.assertAlmostEqual(f, 0.0, places=10)

    def test_von_neumann_entropy(self):
        """Test von Neumann entropy."""
        # Pure state: entropy = 0
        e = von_neumann_entropy(np.array([1, 0]))
        self.assertAlmostEqual(e, 0.0, places=10)

        # Maximally mixed 2-qubit: entropy = 2
        e = von_neumann_entropy(np.eye(4) / 4)
        self.assertAlmostEqual(e, 2.0, places=10)

    def test_random_unitary(self):
        """Test random unitary generation."""
        U = random_unitary(4)
        self.assertTrue(is_unitary(U))
        self.assertEqual(U.shape, (4, 4))


class TestAlgorithms(unittest.TestCase):
    """Test quantum algorithm implementations."""

    def test_qft_matrix(self):
        """Test QFT matrix."""
        from quantumflow.algorithms.qft import qft_matrix
        Q = qft_matrix(3)
        self.assertTrue(is_unitary(Q))
        self.assertEqual(Q.shape, (8, 8))

    def test_grover_success_probability(self):
        """Test Grover's success probability formula."""
        from quantumflow.algorithms.grover import GroverSearch
        grover = GroverSearch(n_qubits=3, marked_states=['101'])
        prob = grover.success_probability()
        # With optimal_iterations using floor formula, R=1 gives P~0.78
        self.assertGreater(prob, 0.7)

    def test_hamiltonian_creation(self):
        """Test Hamiltonian creation."""
        from quantumflow.algorithms.vqe import Hamiltonian, PauliTerm
        H = Hamiltonian(2, [PauliTerm(1.0, "ZZ"), PauliTerm(0.5, "XX")])
        self.assertEqual(H.n_terms, 2)
        self.assertTrue(is_hermitian(H.matrix()))

    def test_hwe_ansatz(self):
        """Test Hardware-Efficient Ansatz."""
        from quantumflow.algorithms.vqe import HWEAnsatz
        ansatz = HWEAnsatz(3, n_layers=2)
        params = np.random.uniform(-np.pi, np.pi, ansatz.n_params())
        circuit = ansatz.construct_circuit(params)
        self.assertIsNotNone(circuit)

    def test_maxcut_qaoa(self):
        """Test MaxCut QAOA."""
        from quantumflow.algorithms.qaoa import MaxCutQAOA
        edges = [(0, 1), (1, 2), (2, 0)]
        maxcut = MaxCutQAOA(edges, n_nodes=3, p=1)
        self.assertIsNotNone(maxcut.cost_hamiltonian)
        cut = maxcut.cut_value('101')
        # Triangle graph: '101' => edges (0,1) and (1,2) cross, (2,0) does not
        self.assertEqual(cut, 2)


class TestNoise(unittest.TestCase):
    """Test noise models and error mitigation."""

    def test_depolarizing_channel(self):
        """Test depolarizing channel."""
        from quantumflow.noise.error_channels import DepolarizingChannel
        ch = DepolarizingChannel(0.1)
        self.assertTrue(ch.is_cptp())

    def test_amplitude_damping(self):
        """Test amplitude damping channel."""
        from quantumflow.noise.error_channels import AmplitudeDampingChannel
        ch = AmplitudeDampingChannel(0.2)
        self.assertTrue(ch.is_cptp())

    def test_noise_model(self):
        """Test noise model."""
        from quantumflow.noise.noise_model import NoiseModel, NoiseConfig
        config = NoiseConfig(single_gate_error=0.01)
        nm = NoiseModel(config)
        self.assertIsNotNone(nm)

    def test_zne(self):
        """Test Zero Noise Extrapolation."""
        from quantumflow.noise.error_mitigation import ZeroNoiseExtrapolation
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0])
        zne._noisy_results = [1.0, 0.8, 0.6]
        result = zne.mitigate(None, noisy_expectations=[1.0, 0.8, 0.6])
        self.assertIsNotNone(result['mitigated_value'])

    def test_measurement_mitigation(self):
        """Test measurement error mitigation."""
        from quantumflow.noise.error_mitigation import MeasurementErrorMitigation
        mem = MeasurementErrorMitigation(n_qubits=2)
        cm = MeasurementErrorMitigation.create_confusion_matrix(2, {0: 0.01, 1: 0.01})
        mem.calibrate(confusion_matrix=cm)
        counts = {'00': 900, '01': 50, '10': 30, '11': 20}
        result = mem.mitigate(counts)
        self.assertIn('mitigated_counts', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
