"""
QuantumFlow Examples and Demos
===============================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_bell_state():
    """Create and analyze a Bell state."""
    print("=" * 60)
    print("Example 1: Bell State (EPR Pair)")
    print("=" * 60)

    from quantumflow import QuantumCircuit, StatevectorSimulator

    # Create Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Draw circuit
    from quantumflow.visualization import CircuitDrawer
    drawer = CircuitDrawer(qc)
    print("\nCircuit:")
    print(drawer.draw_text())

    # Simulate
    sim = StatevectorSimulator()
    result = sim.run(qc, shots=1000)

    print(f"\nMeasurement counts: {result.get_counts()}")
    print(f"Probabilities: {result.get_probabilities()}")

    # Verify entanglement
    U = qc.to_unitary()
    bell_state = U @ np.array([1, 0, 0, 0])
    print(f"\nBell state: {bell_state}")
    print(f"Expected: (|00> + |11>) / sqrt(2)")


def example_grover_search():
    """Demonstrate Grover's search algorithm."""
    print("\n" + "=" * 60)
    print("Example 2: Grover's Search Algorithm")
    print("=" * 60)

    from quantumflow.algorithms.grover import GroverSearch

    # Search for state |101> in 3-qubit space
    grover = GroverSearch(n_qubits=3, marked_states=['101'])
    print(f"\nSearching for: |{'>'.join(grover.marked_states)}> in {2**3} items")
    print(f"Optimal iterations: {grover.optimal_iterations()}")
    print(f"Success probability: {grover.success_probability():.4f}")

    circuit = grover.construct_circuit()
    print(f"\nCircuit depth: {circuit.depth()}")
    print(f"Circuit size: {circuit.size()}")


def example_qft():
    """Demonstrate Quantum Fourier Transform."""
    print("\n" + "=" * 60)
    print("Example 3: Quantum Fourier Transform")
    print("=" * 60)

    from quantumflow.algorithms.qft import QFT, qft_matrix

    # Verify QFT unitarity
    Q = qft_matrix(3)
    print(f"\nQFT matrix (3 qubits) shape: {Q.shape}")

    # Create QFT circuit
    qft = QFT(3)
    circuit = qft.construct_circuit()
    print(f"QFT circuit depth: {circuit.depth()}")
    print(f"QFT gate count: {qft.gate_count()}")

    # Approximate QFT
    aqft = QFT(10, approximation_degree=5)
    print(f"\nApproximate QFT (10 qubits, degree 5)")
    print(f"Exact gates: {qft.gate_count()}")
    print(f"Approximate gates: {aqft.gate_count()}")
    print(f"Reduction: {100 * (1 - aqft.gate_count() / qft.gate_count()):.1f}%")


def example_vqe():
    """Demonstrate Variational Quantum Eigensolver."""
    print("\n" + "=" * 60)
    print("Example 4: Variational Quantum Eigensolver (VQE)")
    print("=" * 60)

    from quantumflow.algorithms.vqe import (
        VQE, Hamiltonian, HWEAnsatz, VQEResult,
    )

    # Transverse field Ising model
    H = Hamiltonian.transverse_field_ising(2, j=1.0, h=-1.0)
    print(f"\nHamiltonian: Transverse Field Ising Model (2 sites)")
    print(f"Number of Pauli terms: {H.n_terms}")
    print(f"H matrix:\n{H.matrix()}")

    # Classical reference energy
    eigenvalues = np.linalg.eigvalsh(H.matrix())
    print(f"\nExact ground state energy: {eigenvalues[0]:.6f}")

    # Set up VQE
    ansatz = HWEAnsatz(2, n_layers=3, rotation_set=['ry', 'rz'])
    print(f"\nAnsatz: Hardware Efficient ({ansatz.n_layers} layers)")
    print(f"Number of parameters: {ansatz.n_params()}")

    vqe = VQE(H, ansatz, optimizer='COBYLA')
    print(f"\nRunning VQE optimization...")

    # Quick optimization with few iterations for demo
    params = np.random.uniform(-np.pi, np.pi, ansatz.n_params())
    circuit = ansatz.construct_circuit(params)
    energy = vqe.energy(params)
    print(f"Initial energy (random params): {energy:.6f}")

    result = vqe.run(max_iterations=50, convergence_threshold=1e-4)
    print(f"\nVQE Results:")
    print(f"  Optimal energy: {result.optimal_energy:.6f}")
    print(f"  Exact energy:   {eigenvalues[0]:.6f}")
    print(f"  Error:          {abs(result.optimal_energy - eigenvalues[0]):.6f}")
    print(f"  Iterations:     {result.iteration_count}")


def example_qaoa_maxcut():
    """Demonstrate QAOA for MaxCut."""
    print("\n" + "=" * 60)
    print("Example 5: QAOA for MaxCut Problem")
    print("=" * 60)

    from quantumflow.algorithms.qaoa import MaxCutQAOA

    # Triangle graph
    edges = [(0, 1), (1, 2), (2, 0)]
    n_nodes = 3

    print(f"\nGraph: Triangle (3 nodes, 3 edges)")
    print(f"Edges: {edges}")
    print(f"Expected max cut: 2 (cut any edge)")

    maxcut = MaxCutQAOA(edges, n_nodes=n_nodes, p=2)
    print(f"\nQAOA depth: {maxcut.p}")
    print(f"Cost Hamiltonian terms: {len(maxcut.cost_hamiltonian.terms)}")

    # Verify cut values
    print("\nAll possible cuts:")
    for bs in ['000', '001', '010', '011', '100', '101', '110', '111']:
        cut = maxcut.cut_value(bs)
        set_0, set_1 = maxcut.get_cut(bs)
        print(f"  {bs}: cut={cut}, partition=({set_0}, {set_1})")


def example_noise_mitigation():
    """Demonstrate noise modeling and error mitigation."""
    print("\n" + "=" * 60)
    print("Example 6: Noise Modeling & Error Mitigation")
    print("=" * 60)

    from quantumflow.noise.error_channels import (
        DepolarizingChannel, AmplitudeDampingChannel,
        ThermalRelaxationChannel,
    )
    from quantumflow.noise.error_mitigation import (
        ZeroNoiseExtrapolation, MeasurementErrorMitigation,
    )

    # Test error channels
    print("\nError Channels:")
    for name, ch in [
        ("Depolarizing (p=0.1)", DepolarizingChannel(0.1)),
        ("Amplitude Damping (g=0.2)", AmplitudeDampingChannel(0.2)),
        ("Thermal Relaxation", ThermalRelaxationChannel(t1=50e-6, t2=30e-6, gate_time=100e-9)),
    ]:
        is_cptp = ch.is_cptp()
        n_kraus = len(ch.kraus_operators())
        print(f"  {name}: CPTP={is_cptp}, Kraus ops={n_kraus}")

    # ZNE demonstration
    print("\nZero Noise Extrapolation:")
    zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], method='richardson')
    zne._noisy_results = [1.0, 0.85, 0.72]
    result = zne.mitigate(None, noisy_expectations=[1.0, 0.85, 0.72])
    print(f"  Noisy values: {result['noisy_values']}")
    print(f"  Extrapolated (zero noise): {result['mitigated_value']:.4f}")

    # Measurement error mitigation
    print("\nMeasurement Error Mitigation:")
    mem = MeasurementErrorMitigation(n_qubits=2)
    cm = MeasurementErrorMitigation.create_confusion_matrix(2, {0: 0.02, 1: 0.02})
    mem.calibrate(confusion_matrix=cm)
    noisy_counts = {'00': 850, '01': 80, '10': 40, '11': 30}
    result = mem.mitigate(noisy_counts)
    print(f"  Noisy counts: {noisy_counts}")
    print(f"  Mitigated counts: {result['mitigated_counts']}")


def example_hydrogen_molecule():
    """Demonstrate VQE for the H2 molecule."""
    print("\n" + "=" * 60)
    print("Example 7: VQE for H2 Molecule (STO-3G)")
    print("=" * 60)

    from quantumflow.algorithms.vqe import Hamiltonian, HWEAnsatz, VQE

    # H2 Hamiltonian
    H = Hamiltonian.hydrogen_molecule()
    print(f"\nH2 molecule Hamiltonian (STO-3G basis)")
    print(f"Number of qubits: {H.n_qubits}")
    print(f"Number of Pauli terms: {H.n_terms}")

    # Exact solution
    eigenvalues = np.linalg.eigvalsh(H.matrix())
    print(f"\nExact ground state energy: {eigenvalues[0]:.6f} Hartree")

    # VQE
    ansatz = HWEAnsatz(H.n_qubits, n_layers=2, rotation_set=['ry', 'rz'])
    vqe = VQE(H, ansatz, optimizer='COBYLA')

    result = vqe.run(max_iterations=30, convergence_threshold=1e-4)
    print(f"\nVQE ground state energy: {result.optimal_energy:.6f} Hartree")
    print(f"Error from exact: {abs(result.optimal_energy - eigenvalues[0]):.6f} Hartree")
    print(f"Chemical accuracy (1.6 mHa): {'YES' if abs(result.optimal_energy - eigenvalues[0]) < 0.0016 else 'NO'}")


def example_utils():
    """Demonstrate utility functions."""
    print("\n" + "=" * 60)
    print("Example 8: Quantum Utility Functions")
    print("=" * 60)

    from quantumflow.utils.math import (
        fidelity, trace_distance, purity, von_neumann_entropy,
        state_to_bloch, bloch_to_state, random_density_matrix,
        partial_trace,
    )

    # State comparison
    psi1 = np.array([1, 0], dtype=np.complex128)
    psi2 = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    psi3 = np.array([1, 0], dtype=np.complex128)

    print(f"\nFidelity(|0>, |0>): {fidelity(psi1, psi3):.6f}")
    print(f"Fidelity(|0>, |+>): {fidelity(psi1, psi2):.6f}")
    print(f"Trace distance(|0>, |+>): {trace_distance(psi1, psi2):.6f}")

    # Entropy
    dm_pure = np.outer(psi1, psi1.conj())
    dm_mixed = np.eye(2) / 2
    dm_maximally_mixed = np.eye(4) / 4

    print(f"\nPurity(|0><0|): {purity(dm_pure):.6f}")
    print(f"Purity(I/2): {purity(dm_mixed):.6f}")
    print(f"Purity(I/4): {purity(dm_maximally_mixed):.6f}")

    print(f"\nVon Neumann entropy(|0><0|): {von_neumann_entropy(dm_pure):.6f}")
    print(f"Von Neumann entropy(I/2): {von_neumann_entropy(dm_mixed):.6f}")
    print(f"Von Neumann entropy(I/4): {von_neumann_entropy(dm_maximally_mixed):.6f}")

    # Bloch sphere
    bloch = state_to_bloch(psi2)
    print(f"\nBloch vector of |+>: {bloch}")
    recovered = bloch_to_state(bloch)
    print(f"Recovered state fidelity: {fidelity(psi2, recovered):.6f}")

    # Random density matrix
    rho = random_density_matrix(2)
    print(f"\nRandom density matrix:")
    print(f"  Purity: {purity(rho):.6f}")
    print(f"  Entropy: {von_neumann_entropy(rho):.6f}")


if __name__ == '__main__':
    examples = [
        example_bell_state,
        example_grover_search,
        example_qft,
        example_vqe,
        example_qaoa_maxcut,
        example_noise_mitigation,
        example_hydrogen_molecule,
        example_utils,
    ]

    print("QuantumFlow — Advanced Quantum Computing Framework")
    print("=" * 60)
    print(f"Running {len(examples)} examples...\n")

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n[ERROR] {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
