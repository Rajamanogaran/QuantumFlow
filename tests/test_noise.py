"""
Noise model and error mitigation tests
======================================

Regression coverage for the 0.2 noise fixes: ``NoiseModel.apply_noise``
previously inserted nothing (circuits lacked ``append_kraus``), the
depolarizing Kraus set was not a physical channel, and Richardson ZNE
dropped the ``(0 - x_j)`` numerator terms from the Lagrange weights.
"""

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.core.operation import KrausChannel
from quantumflow.noise.error_mitigation import ZeroNoiseExtrapolation
from quantumflow.noise.noise_model import NoiseConfig, NoiseModel


def _x_circuit():
    qc = qf.QuantumCircuit(2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    return qc


class TestKrausChannel:
    def test_valid_channel_accepted(self):
        ch = KrausChannel(
            [np.sqrt(0.9) * np.eye(2), np.sqrt(0.1) * np.array([[0, 1], [1, 0]])],
            [0],
        )
        assert ch.num_qubits == 1
        assert len(ch.kraus_ops) == 2

    def test_incomplete_channel_rejected(self):
        with pytest.raises(ValueError):
            KrausChannel([np.sqrt(0.5) * np.eye(2), np.sqrt(0.4) * np.eye(2)], [0])

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ValueError):
            KrausChannel([np.eye(2)], [0, 1])

    def test_circuit_append_and_simulate(self):
        qc = qf.QuantumCircuit(1)
        qc.x(0)
        qc.append_kraus(
            [np.sqrt(0.5) * np.eye(2), np.sqrt(0.5) * np.array([[0, 1], [1, 0]])],
            [0],
        )
        qc.measure([0], [0])
        counts = qf.DensityMatrixSimulator().run(qc, shots=4000).get_counts()
        # 50/50 X-flipped mixture: both outcomes appear substantially
        p1 = counts.get("1", 0) / 4000
        assert 0.2 < p1 < 0.8

    def test_statevector_simulator_rejects_kraus(self):
        qc = qf.QuantumCircuit(1)
        qc.append_kraus([np.eye(2)], [0])
        with pytest.raises(TypeError):
            qf.StatevectorSimulator().run(qc, shots=10)


class TestNoiseModel:
    def test_apply_noise_inserts_channels(self):
        """Regression: apply_noise was a silent no-op."""
        nm = NoiseModel(NoiseConfig(single_gate_error=0.05, two_gate_error=0.1))
        noisy = nm.apply_noise(_x_circuit())
        assert len(noisy.data) > len(_x_circuit().data)

    def test_depolarizing_kraus_is_complete(self):
        """Regression: the old Kraus set was not a physical channel."""
        for n in (1, 2):
            kraus = NoiseModel._depolarizing_kraus(0.1, n)
            comp = sum(k.conj().T @ k for k in kraus)
            assert np.allclose(comp, np.eye(2**n), atol=1e-9)

    def test_noisy_simulations_degrade_with_scale(self):
        qc = _x_circuit()
        nm = NoiseModel(
            NoiseConfig(single_gate_error=0.05, two_gate_error=0.1)
        )
        probs = []
        for scale in (1.0, 3.0):
            circ = nm.apply_noise(qc, noise_scale=scale)
            counts = qf.DensityMatrixSimulator().run(circ, shots=4000).get_counts()
            probs.append(counts.get("10", 0) / 4000)
        # P(10) is 1 without noise; stronger noise degrades it further
        assert probs[0] < 0.999
        assert probs[1] < probs[0] - 0.05

    def test_attached_noise_model_affects_results(self):
        nm = NoiseModel(NoiseConfig(single_gate_error=0.2))
        sim = qf.DensityMatrixSimulator(noise_model=nm)
        clean = qf.StatevectorSimulator().run(_x_circuit(), shots=4000)
        p_clean = clean.get_counts().get("10", 0) / 4000
        p_noisy = sim.run(_x_circuit(), shots=4000).get_counts().get("10", 0) / 4000
        assert p_noisy < p_clean - 0.05

    def test_embed_scatters_correctly(self):
        # X on qubit 1 (LSB) embedded into 2 qubits must flip the LSB
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        full = NoiseModel._embed(x, [1], 2)
        expected = np.kron(np.eye(2), x)
        assert np.allclose(full, expected)


class TestZeroNoiseExtrapolation:
    def test_richardson_exact_on_quadratics(self):
        """Regression: weights dropped the (0 - x_j) numerators."""
        zne = ZeroNoiseExtrapolation([1.0, 2.0, 3.0])
        out = zne.mitigate({}, noisy_expectations=[0.9, 0.7, 0.4])
        assert abs(out["mitigated_value"] - 1.0) < 1e-9

    def test_linear_extrapolation_sensible(self):
        zne = ZeroNoiseExtrapolation([1.0, 2.0, 3.0], method="linear")
        out = zne.mitigate({}, noisy_expectations=[0.88, 0.79, 0.73])
        assert 0.8 < out["mitigated_value"] <= 1.0

    def test_extrapolation_beats_noisy_value(self):
        zne = ZeroNoiseExtrapolation([1.0, 2.0, 3.0], method="linear")
        out = zne.mitigate({}, noisy_expectations=[0.88, 0.79, 0.73])
        assert out["mitigated_value"] > max(0.88, 0.79, 0.73)
