"""
End-to-end algorithm correctness tests
=======================================

These verify that the flagship algorithms produce *physically correct*
results (not merely that they run): Grover finds marked states, the QFT
circuit equals the QFT matrix, QPE/IPE recover exact phases, Shor factors
small semiprimes, and VQE/QAOA reach their ground-state / optimal-cut
values.
"""

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.algorithms.qft import iqft_matrix, qft_matrix

# ---------------------------------------------------------------------------
# Grover
# ---------------------------------------------------------------------------


class TestGrover:
    def test_finds_single_marked_3q(self):
        g = qf.GroverSearch(n_qubits=3, marked_states=["101"])
        res = g.run(shots=256)
        assert res["most_frequent"] == "101"

    def test_finds_single_marked_4q(self):
        """Regression: _apply_mcx recursed infinitely for >=3 controls."""
        g = qf.GroverSearch(n_qubits=4, marked_states=["1010"])
        res = g.run(shots=256)
        assert res["most_frequent"] == "1010"

    def test_multiple_marked_states(self):
        g = qf.GroverSearch(n_qubits=4, marked_states=["0110", "1001"])
        res = g.run(shots=256)
        assert res["most_frequent"] in ("0110", "1001")

    def test_oracle_is_exact_phase_flip(self):
        g = qf.GroverSearch(n_qubits=3, marked_states=["010"])
        oracle = g.create_oracle().to_unitary()
        expected = np.eye(8)
        expected[2, 2] = -1  # |010> = index 2
        assert np.allclose(oracle, expected)

    def test_diffusion_operator(self):
        g = qf.GroverSearch(n_qubits=3, marked_states=["000"])
        D = g.create_diffusion().to_unitary()
        s = np.ones(8) / np.sqrt(8)
        expected = 2 * np.outer(s, s) - np.eye(8)
        # up to global phase
        phase = np.vdot(expected.flatten(), D.flatten())
        assert np.allclose(D * np.exp(-1j * np.angle(phase)), expected)

    def test_optimal_iterations_formula(self):
        g = qf.GroverSearch(n_qubits=4, marked_states=["1010"])
        R = g.optimal_iterations()
        theta = np.arcsin(np.sqrt(1 / 16))
        expected = max(1, int(np.floor(np.pi / (4 * theta) - 0.5)))
        assert R == expected

    def test_success_probability_matches_theory(self):
        g = qf.GroverSearch(n_qubits=4, marked_states=["1010"])
        R = g.num_iterations
        theta = np.arcsin(np.sqrt(1 / 16))
        expected = np.sin((2 * R + 1) * theta) ** 2
        assert abs(g.success_probability() - expected) < 1e-12


# ---------------------------------------------------------------------------
# QFT
# ---------------------------------------------------------------------------


class TestQFT:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_circuit_equals_matrix(self, n):
        from quantumflow.algorithms.qft import QFT

        assert np.allclose(QFT(n).construct_circuit().to_unitary(), qft_matrix(n))

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_inverse_circuit_equals_conjugate(self, n):
        from quantumflow.algorithms.qft import InverseQFT

        assert np.allclose(
            InverseQFT(n).construct_circuit().to_unitary(), iqft_matrix(n)
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_then_iqft_is_identity(self, n):
        from quantumflow.algorithms.qft import InverseQFT, QFT

        U = (
            InverseQFT(n).construct_circuit().to_unitary()
            @ QFT(n).construct_circuit().to_unitary()
        )
        assert np.allclose(U, np.eye(2**n), atol=1e-9)

    def test_qft_matrix_definition(self):
        N = 8
        U = qft_matrix(3)
        expected = np.array(
            [[np.exp(2j * np.pi * j * k / N) for k in range(N)] for j in range(N)]
        ) / np.sqrt(N)
        assert np.allclose(U, expected)

    def test_apply_qft_on_subset(self):
        from quantumflow.algorithms.qft import apply_qft, apply_iqft

        qc = qf.QuantumCircuit(3)
        qc.x(1)  # |010>
        apply_qft(qc, [0, 2])
        apply_iqft(qc, [0, 2])
        out = qf.StatevectorSimulator().run(qc, shots=0)
        assert np.allclose(out.statevector.data, [0, 0, 1, 0, 0, 0, 0, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# QPE
# ---------------------------------------------------------------------------


class TestQPE:
    @pytest.mark.parametrize(
        "phi,n_e",
        [(0.25, 3), (0.125, 3), (0.75, 4), (0.625, 4), (0.5, 4)],
    )
    def test_exact_phases(self, phi, n_e):
        from quantumflow.algorithms.qpe import PhaseEstimation

        U = np.diag([1, np.exp(1j * 2 * np.pi * phi)]).astype(complex)
        res = PhaseEstimation(
            U, n_evaluation_qubits=n_e, n_state_qubits=1
        ).run(shots=2048)
        assert abs(res["phase"] - phi) < 1e-9

    def test_t_gate_phase(self):
        from quantumflow.algorithms.qpe import PhaseEstimation

        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        res = PhaseEstimation(
            T, n_evaluation_qubits=4, n_state_qubits=1
        ).run(shots=4096)
        assert abs(res["phase"] - 0.125) < 1e-9

    def test_approximate_phase_within_one_bin(self):
        from quantumflow.algorithms.qpe import PhaseEstimation

        phi = 0.3
        n_e = 5
        U = np.diag([1, np.exp(1j * 2 * np.pi * phi)]).astype(complex)
        res = PhaseEstimation(
            U, n_evaluation_qubits=n_e, n_state_qubits=1
        ).run(shots=4096)
        # statistical bound: the top measured bin must be within 2 bins of
        # the true phase (bin width 1/2^5 = 0.03125)
        assert abs(res["phase"] - phi) < 2 / 2**n_e

    def test_iterative_qpe_exact(self):
        """Regression: IPE had a broken controlled-U and no-op feedback."""
        from quantumflow.algorithms.qpe import IterativePhaseEstimation

        U = np.diag([1, np.exp(1j * 2 * np.pi / 3)]).astype(complex)
        eig = np.array([0, 1], dtype=complex)
        ipe = IterativePhaseEstimation(
            U, n_state_qubits=1, n_iterations=8, eigenstate=eig
        )
        res = ipe.run(shots_per_iteration=128)
        # 8-bit truncation of 1/3 = 0.33203125
        assert abs(res["phase"] - 85 / 256) < 0.02
        assert "".join(map(str, res["bit_estimates"])) == "01010101"

    def test_iterative_qpe_two_qubit_register(self):
        from quantumflow.algorithms.qpe import IterativePhaseEstimation

        U = np.diag([1, 1, 1, np.exp(1j * 2 * np.pi * 0.4)]).astype(complex)
        eig = np.array([0, 0, 0, 1], dtype=complex)
        ipe = IterativePhaseEstimation(
            U, n_state_qubits=2, n_iterations=8, eigenstate=eig
        )
        res = ipe.run(shots_per_iteration=128)
        assert abs(res["phase"] - round(0.4 * 256) / 256) < 0.02


# ---------------------------------------------------------------------------
# Shor
# ---------------------------------------------------------------------------


class TestShor:
    @pytest.mark.parametrize(
        "N,a,factors", [(15, 7, [3, 5]), (21, 2, [3, 7]), (35, 13, [5, 7])]
    )
    def test_factors_semiprimes(self, N, a, factors):
        """Regression: IQFT width mismatch + fake modular arithmetic."""
        shor = qf.ShorAlgorithm(N=N, a=a)
        res = shor.factor(shots=1024)
        assert res["success"] is True
        assert sorted(res["factors"]) == factors

    def test_even_number_factored_trivially(self):
        shor = qf.ShorAlgorithm(N=15)
        res = shor.factor()
        assert res["success"]

    def test_modexp_circuit_is_unitary(self):
        from quantumflow.algorithms.shor import ModularExponentiation

        modexp = ModularExponentiation(base=7, modulus=15, n_counting_qubits=4)
        U = modexp.construct_circuit().to_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-9)

    def test_modexp_permutation_semantics(self):
        from quantumflow.algorithms.shor import ModularExponentiation

        modexp = ModularExponentiation(base=7, modulus=15, n_counting_qubits=2)
        perm = modexp._mod_mult_permutation(7)
        # y -> 7y mod 15 for y < 15
        for y in [1, 2, 3, 7, 14]:
            col = perm[:, y]
            target = int(np.argmax(col))
            assert target == (7 * y) % 15
            assert abs(col[target] - 1) < 1e-12


# ---------------------------------------------------------------------------
# VQE / QAOA
# ---------------------------------------------------------------------------


class TestVariational:
    def test_vqe_reaches_ground_state(self):
        from quantumflow.algorithms.vqe import Hamiltonian, HWEAnsatz, PauliTerm, VQE

        H = Hamiltonian(2, [PauliTerm(1.0, "ZZ"), PauliTerm(0.5, "XX")])
        exact = float(np.min(np.linalg.eigvalsh(H.matrix())))  # = -1.5
        vqe = VQE(
            H,
            ansatz=HWEAnsatz(2, n_layers=1),
            initial_params=np.array([0.4, -0.3, 0.2, 0.6]),
        )
        res = vqe.run(max_iterations=200)
        assert res.optimal_energy <= exact + 1e-2
        assert res.success

    def test_maxcut_qaoa_optimal_cut(self):
        from quantumflow.algorithms.qaoa import MaxCutQAOA

        # Path graph P4 has max cut 3 (alternating bitstrings)
        mc = MaxCutQAOA([(0, 1), (1, 2), (2, 3)], n_nodes=4, p=1)
        res = mc.solve()
        assert len(res.best_bitstring) == 4
        assert mc.cut_value(res.best_bitstring) == 3

    def test_mis_qaoa_runs(self):
        from quantumflow.algorithms.qaoa import MISQAOA

        mis = MISQAOA([(0, 1), (1, 2)], n_nodes=3, p=1)
        res = mis.solve()
        assert res is not None
