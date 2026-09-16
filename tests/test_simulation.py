"""
Simulation-layer regression tests
=================================

Locks in the correctness fixes from the 0.2.0 audit: gate embedding
conventions, measurement semantics, per-shot execution, the gate cache,
``compose``/``measure`` APIs and ``Statevector`` arithmetic.
"""

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import Measurement
from quantumflow.simulation.simulator import (
    DensityMatrixSimulator,
    StatevectorSimulator,
)

X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


# ---------------------------------------------------------------------------
# Gate embedding: to_unitary must agree with the simulator
# ---------------------------------------------------------------------------

class TestGateEmbedding:
    @pytest.mark.parametrize(
        "build",
        [
            lambda qc: qc.x(2),
            lambda qc: qc.cx(0, 2),
            lambda qc: qc.cx(2, 0),
            lambda qc: qc.swap(0, 2),
            lambda qc: qc.ccx(0, 1, 2),
            lambda qc: qc.ccx(2, 1, 0),
            lambda qc: (qc.h(1), qc.cx(1, 2)),
            lambda qc: qc.crz(0.7, 0, 2),
            lambda qc: qc.cp(1.1, 2, 0),
            lambda qc: qc.mcz([0, 1], 2),
        ],
        ids=lambda f: getattr(f, "__name__", "case"),
    )
    def test_to_unitary_matches_simulator(self, build):
        qc = QuantumCircuit(3)
        build(qc)
        U = qc.to_unitary()
        rng = np.random.default_rng(7)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        out = StatevectorSimulator().run(qc, shots=0, initial_state=psi)
        assert np.allclose(out.statevector.data, U @ psi, atol=1e-9)

    def test_unitarity_of_random_circuit(self):
        rng = np.random.default_rng(3)
        qc = QuantumCircuit(4)
        for _ in range(12):
            qc.rz(rng.uniform(0, 2 * np.pi), rng.integers(0, 4))
            qc.h(rng.integers(0, 4))
        qc.cx(2, 0)
        qc.swap(0, 3)
        U = qc.to_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-9)

    def test_embed_single_qubit_msb(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        assert np.allclose(qc.to_unitary(), np.kron(X, np.eye(2)))

    def test_embed_single_qubit_lsb(self):
        qc = QuantumCircuit(2)
        qc.x(1)
        assert np.allclose(qc.to_unitary(), np.kron(np.eye(2), X))


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

class TestMeasurement:
    def test_marginal_over_multiple_qubits(self):
        """Regression: axis ordering bug in apply_measurement."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])
        res = StatevectorSimulator().run(qc, shots=200)
        counts = res.get_counts()
        assert set(counts) <= {"000", "111"}

    def test_measure_auto_creates_classical_register(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure([0, 1], [0, 1])
        assert qc.num_clbits == 2

    def test_measure_auto_creates_growing_register(self):
        qc = QuantumCircuit(2)
        qc.measure([0], [3])  # cbit index beyond current size
        assert qc.num_clbits == 4

    def test_per_shot_semantics_mid_circuit(self):
        """Each shot must see a fresh collapse (regression for
        collapse-once-then-sample)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(Measurement(), [0])
        res = StatevectorSimulator().run(qc, shots=400)
        counts = res.get_counts()
        assert set(counts) <= {"0", "1"}
        # a fair coin: neither outcome may dominate pathologically
        assert 100 < counts.get("0", 0) < 300

    def test_bell_partial_measurement_statistics(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(Measurement(), [0])
        res = StatevectorSimulator().run(qc, shots=500)
        counts = res.get_counts()
        assert set(counts) <= {"0", "1"}
        assert 150 < counts.get("0", 0) < 350

    def test_density_matrix_partial_measurement(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(Measurement(), [0])
        res = DensityMatrixSimulator().run(qc, shots=200)
        assert set(res.get_counts()) <= {"0", "1"}

    def test_counts_only_measured_qubits(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.append(Measurement(), [0])
        qc.append(Measurement(), [2])
        res = StatevectorSimulator().run(qc, shots=100)
        for bits in res.get_counts():
            assert len(bits) == 2  # qubits 0 and 2 only

    def test_density_matrix_matches_statevector(self):
        qc = QuantumCircuit(3)
        qc.h(1)
        qc.cx(2, 0)
        qc.cp(1.3, 0, 1)
        qc.t(2)
        sv = StatevectorSimulator().run(qc, shots=0).statevector
        dm = DensityMatrixSimulator().run(qc, shots=0).density_matrix
        assert np.allclose(sv.to_density_matrix().data, dm.data, atol=1e-9)


# ---------------------------------------------------------------------------
# gate cache identity
# ---------------------------------------------------------------------------

class TestGateCache:
    def test_same_named_unitary_gates_do_not_collide(self):
        """Regression: cache keyed by name returned the first matrix."""
        qc = QuantumCircuit(1)
        qc.append(qf.UnitaryGate(X, name="unitary"), [0])
        S = np.diag([1, 1j]).astype(complex)
        qc.append(qf.UnitaryGate(S, name="unitary"), [0])
        out = StatevectorSimulator().run(
            qc, shots=0, initial_state=np.array([1, 0], dtype=complex)
        )
        expected = S @ X @ np.array([1, 0], dtype=complex)
        assert np.allclose(out.statevector.data, expected)

    def test_density_matrix_same_named_gates(self):
        qc = QuantumCircuit(1)
        qc.append(qf.UnitaryGate(X, name="g"), [0])
        qc.append(qf.UnitaryGate(np.diag([1, 1j]).astype(complex), name="g"), [0])
        out = DensityMatrixSimulator().run(qc, shots=0)
        expected = np.diag([0, 1]).astype(complex)
        assert np.allclose(out.density_matrix.data, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# compose / measure API
# ---------------------------------------------------------------------------

class TestComposeAPI:
    # CX with control on the most significant qubit
    CX01 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )

    def test_compose_identity_mapping(self):
        a = QuantumCircuit(2)
        a.h(0)
        b = QuantumCircuit(2)
        b.cx(0, 1)
        c = a.compose(b)
        expected = self.CX01 @ np.kron(H, np.eye(2))
        assert np.allclose(c.to_unitary(), expected)

    def test_compose_with_qubit_mapping(self):
        a = QuantumCircuit(3)
        b = QuantumCircuit(1)
        b.x(0)
        a.compose(b, qubits=[2], inplace=True)
        assert np.allclose(a.to_unitary(), np.kron(np.eye(4), X))

    def test_compose_non_contiguous_mapping(self):
        c = QuantumCircuit(3)
        c.h(0)
        d = QuantumCircuit(2)
        d.cx(0, 1)
        c.compose(d, qubits=[0, 2], inplace=True)
        # Bell pair between qubits 0 (control, MSB) and 2 (target, LSB)
        CX02 = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            j = i ^ 0b001 if (i & 0b100) else i
            CX02[j, i] = 1
        expected = CX02 @ np.kron(H, np.eye(4))
        assert np.allclose(c.to_unitary(), expected, atol=1e-9)

    def test_compose_mapping_validation(self):
        a = QuantumCircuit(3)
        b = QuantumCircuit(2)
        with pytest.raises(ValueError):
            a.compose(b, qubits=[0])            # wrong length
        with pytest.raises(ValueError):
            a.compose(b, qubits=[0, 0])         # duplicates
        with pytest.raises(ValueError):
            a.compose(b, qubits=[0, 9])         # out of range


# ---------------------------------------------------------------------------
# Statevector arithmetic and to_circuit
# ---------------------------------------------------------------------------

class TestStatevectorAPI:
    def test_scalar_arithmetic_returns_statevector(self):
        """Scalar ops return a Statevector and keep it a valid unit state."""
        sv = qf.Statevector([1, 0, 0, 1]) / np.sqrt(2)
        assert isinstance(sv, qf.Statevector)
        np.testing.assert_allclose(sv.data, [0.70710678, 0, 0, 0.70710678], atol=1e-9)
        sv2 = 2 * sv
        assert isinstance(sv2, qf.Statevector)
        sv3 = sv * 0.5
        assert isinstance(sv3, qf.Statevector)
        # normalization is preserved (results stay physical states)
        for s in (sv2, sv3):
            assert abs(np.linalg.norm(s.data) - 1.0) < 1e-12

    def test_to_circuit_basis_state(self):
        sv = qf.Statevector.from_label("101")
        circ = sv.to_circuit()
        out = StatevectorSimulator().run(circ, shots=0)
        np.testing.assert_allclose(out.statevector.data, sv.data, atol=1e-12)

    def test_to_circuit_bell_state_fidelity(self):
        sv = qf.Statevector([1, 0, 0, 1]) / np.sqrt(2)
        circ = sv.to_circuit()
        out = StatevectorSimulator().run(circ, shots=0)
        fid = abs(np.vdot(sv.data, out.statevector.data)) ** 2
        assert fid > 1 - 1e-9

    def test_to_circuit_random_state_fidelity(self):
        rng = np.random.default_rng(11)
        raw = rng.normal(size=16) + 1j * rng.normal(size=16)
        sv = qf.Statevector(raw)
        circ = sv.to_circuit()
        out = StatevectorSimulator().run(circ, shots=0)
        fid = abs(np.vdot(sv.data, out.statevector.data)) ** 2
        assert fid > 1 - 1e-9
