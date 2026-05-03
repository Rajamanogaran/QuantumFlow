"""
Quantum Algorithms Module
=========================

Implements major quantum algorithms including Grover's search,
Shor's factoring, Quantum Fourier Transform, Phase Estimation,
Variational Quantum Eigensolver (VQE), and QAOA.
"""

from quantumflow.algorithms.grover import GroverSearch, AmplitudeAmplification
from quantumflow.algorithms.qft import QFT, InverseQFT, QuantumAdder, QuantumMultiplier
from quantumflow.algorithms.shor import ShorAlgorithm, ModularExponentiation
from quantumflow.algorithms.qpe import PhaseEstimation, IterativePhaseEstimation
from quantumflow.algorithms.vqe import VQE, Hamiltonian, UCCSDAnsatz, HWEAnsatz
from quantumflow.algorithms.qaoa import QAOA, MaxCutQAOA, MISQAOA, TSPQAOA

__all__ = [
    "GroverSearch", "AmplitudeAmplification",
    "QFT", "InverseQFT", "QuantumAdder", "QuantumMultiplier",
    "ShorAlgorithm", "ModularExponentiation",
    "PhaseEstimation", "IterativePhaseEstimation",
    "VQE", "Hamiltonian", "UCCSDAnsatz", "HWEAnsatz",
    "QAOA", "MaxCutQAOA", "MISQAOA", "TSPQAOA",
]
