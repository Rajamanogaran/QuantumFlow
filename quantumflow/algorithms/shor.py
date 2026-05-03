"""
Shor's Factoring Algorithm
==========================

Implements Shor's algorithm for integer factorization.

Shor's algorithm can factor integers in polynomial time on a quantum computer,
providing an exponential speedup over the best known classical algorithms.

The algorithm consists of:
1. Classical pre-processing: choose a random coprime a
2. Quantum order finding: find the order r of a mod N
3. Classical post-processing: extract factors from r using continued fractions

References:
    - Shor, P.W. (1997). Polynomial-time algorithms for prime factorization
      and discrete logarithms on a quantum computer.
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from fractions import Fraction

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, CNOTGate, ControlledGate, SwapGate,
        PhaseGate, Measurement, UnitaryGate,
    )
    from quantumflow.core.state import Statevector
    from quantumflow.simulation.simulator import StatevectorSimulator
except ImportError:
    pass


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended GCD: returns (g, x, y) such that a*x + b*y = g."""
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def mod_inverse(a: int, m: int) -> int:
    """Compute modular multiplicative inverse of a mod m."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist for a={a}, m={m}")
    return x % m


def continued_fraction(x: float, max_denominator: int = 10000) -> List[Tuple[int, int]]:
    """
    Compute convergents of continued fraction expansion.

    Parameters
    ----------
    x : float
        Number to expand.
    max_denominator : int
        Maximum denominator for convergents.

    Returns
    -------
    List[Tuple[int, int]]
        List of (numerator, denominator) convergents.
    """
    convergents = []
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    y = x
    while True:
        a = int(math.floor(y))
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev

        if k_next > max_denominator:
            break

        convergents.append((h_next, k_next))
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next

        y = y - a
        if abs(y) < 1e-10:
            break
        y = 1.0 / y

    return convergents


class ModularExponentiation:
    """
    Quantum modular exponentiation circuit.

    Computes |x>|0> -> |x>|a^x mod N> using quantum circuits.

    The implementation uses the standard approach of repeated squaring
    and modular multiplication via quantum phase estimation.

    Parameters
    ----------
    base : int
        The base a in a^x mod N.
    modulus : int
        The modulus N.
    n_counting_qubits : int
        Number of qubits in the counting register.
    """

    def __init__(self, base: int, modulus: int, n_counting_qubits: int) -> None:
        self.base = base % modulus
        self.modulus = modulus
        self.n_counting_qubits = n_counting_qubits
        self.n_work_qubits = math.ceil(math.log2(modulus))

    def construct_circuit(self) -> QuantumCircuit:
        """
        Construct the modular exponentiation circuit.

        Returns
        -------
        QuantumCircuit
            Circuit implementing modular exponentiation.
        """
        n_count = self.n_counting_qubits
        n_work = self.n_work_qubits
        total = n_count + 2 * n_work
        circuit = QuantumCircuit(total)

        count_qubits = list(range(n_count))
        work_qubits = list(range(n_count, n_count + n_work))
        ancilla_qubits = list(range(n_count + n_work, total))

        # Pre-compute a^(2^i) mod N for each counting qubit
        powers = [pow(self.base, 2**i, self.modulus) for i in range(n_count)]

        for i in range(n_count):
            if powers[i] == 1:
                continue  # Identity operation

            # Controlled modular multiplication by a^(2^i)
            self._controlled_mod_mult(circuit, powers[i], count_qubits[i],
                                       work_qubits, ancilla_qubits)

        return circuit

    def _controlled_mod_mult(
        self,
        circuit: QuantumCircuit,
        multiplier: int,
        control: int,
        target: List[int],
        ancilla: List[int],
    ) -> None:
        """Apply controlled modular multiplication."""
        n = len(target)

        if multiplier == 0:
            for q in target:
                circuit.x(q)
            return

        # Encode multiplier in binary and apply controlled additions
        for j in range(n):
            if (multiplier >> j) & 1:
                for k in range(n):
                    circuit.ccx(control, target[j], target[(j + k) % n])

        # Simplified modular reduction (valid for small moduli)
        # For larger moduli, need full quantum modular arithmetic
        if self.modulus < (1 << n):
            mod_bits = math.ceil(math.log2(self.modulus))
            for j in range(mod_bits, n):
                for k in range(mod_bits):
                    if (self.modulus >> k) & 1:
                        circuit.ccx(control, target[j], target[k])


class ShorAlgorithm:
    """
    Shor's Factoring Algorithm.

    Factors a composite integer N using quantum order finding.

    Algorithm overview:
    1. Check if N is even (factor out 2)
    2. For a random coprime a, find the order r of a mod N using quantum circuit
    3. If r is even and a^(r/2) != -1 mod N, then gcd(a^(r/2) - 1, N) and
       gcd(a^(r/2) + 1, N) are non-trivial factors

    Parameters
    ----------
    N : int
        The integer to factor.
    a : Optional[int]
        Base for order finding. If None, chosen randomly.

    Attributes
    ----------
    N : int
        Number to factor.
    factors : List[int]
        Found factors.

    Examples
    --------
    >>> shor = ShorAlgorithm(15)
    >>> result = shor.factor()
    >>> print(result.factors)  # e.g., [3, 5]

    >>> shor = ShorAlgorithm(21, a=2)
    >>> result = shor.factor()
    """

    def __init__(self, N: int, a: Optional[int] = None) -> None:
        if N <= 1:
            raise ValueError("N must be greater than 1")
        if N <= 3:
            raise ValueError(f"{N} is prime (trivially)")
        if N % 2 == 0:
            self._trivial_factor = 2
        else:
            self._trivial_factor = None

        self.N = N
        self.a = a
        self.factors: List[int] = []
        self._order = None

    def is_prime(self, n: int, k: int = 20) -> bool:
        """
        Miller-Rabin primality test.

        Parameters
        ----------
        n : int
        k : int
            Number of rounds.

        Returns
        -------
        bool
            True if n is (probably) prime.
        """
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False

        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        for _ in range(k):
            a = np.random.randint(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def choose_coprime(self) -> int:
        """Choose a random coprime a of N."""
        if self.a is not None:
            g = gcd(self.a, self.N)
            if g > 1 and g < self.N:
                return self.a
            return self.a

        while True:
            a = np.random.randint(2, self.N)
            g = gcd(a, self.N)
            if g == 1:
                return a
            if 1 < g < self.N:
                return a  # Found a factor

    def order_finding(
        self,
        a: int,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 1024,
    ) -> Optional[int]:
        """
        Find the order r of a modulo N using quantum phase estimation.

        Parameters
        ----------
        a : int
            Base for order finding.
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        shots : int
            Number of shots for measurement.

        Returns
        -------
        Optional[int]
            The order r, or None if not found.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        # Number of counting qubits determines precision
        n_count = max(8, 2 * math.ceil(math.log2(self.N)))
        n_work = math.ceil(math.log2(self.N))
        total_qubits = n_count + 2 * n_work

        # Build the order finding circuit
        circuit = QuantumCircuit(total_qubits)
        count_qubits = list(range(n_count))
        work_qubits = list(range(n_count, n_count + n_work))

        # Initialize counting register to |+>
        for q in count_qubits:
            circuit.h(q)

        # Initialize work register to |1>
        circuit.x(work_qubits[0])

        # Apply controlled modular exponentiation
        mod_exp = ModularExponentiation(a, self.N, n_count)
        mod_exp_circuit = mod_exp.construct_circuit()
        circuit.compose(mod_exp_circuit, inplace=True)

        # Apply inverse QFT on counting register
        from quantumflow.algorithms.qft import InverseQFT
        iqft = InverseQFT(n_count)
        iqft_circuit = iqft.construct_circuit()
        # Map IQFT qubits to our counting qubits
        circuit.compose(iqft_circuit, inplace=True)

        # Measure counting register
        for q in count_qubits:
            circuit.append(Measurement(), [q])

        # Run the circuit
        result = simulator.run(circuit, shots=shots)
        counts = result.get_counts()

        # Process measurement results to find the order
        best_r = None
        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            measured = int(bitstring[:n_count][::-1], 2)  # Reverse bit order
            phase = measured / (2 ** n_count)

            if phase == 0:
                continue

            # Use continued fractions to find r
            convergents = continued_fraction(phase, max_denominator=self.N)
            for num, den in convergents:
                if den == 0:
                    continue
                # Check if this denominator is the order
                if pow(a, den, self.N) == 1:
                    if den % 2 == 0:  # Need even order for factoring
                        best_r = den
                        break

            if best_r is not None:
                break

        return best_r

    def extract_factors(self, a: int, r: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract factors from order r.

        Parameters
        ----------
        a : int
            Base.
        r : int
            Order of a mod N.

        Returns
        -------
        Tuple[Optional[int], Optional[int]]
            Pair of factors (p, q), or (None, None) if extraction fails.
        """
        if r is None or r % 2 != 0:
            return None, None

        x = pow(a, r // 2, self.N)
        if x == 1 or x == self.N - 1:
            return None, None

        p = gcd(x - 1, self.N)
        q = gcd(x + 1, self.N)

        if 1 < p < self.N and 1 < q < self.N:
            return p, q
        return None, None

    def construct_circuit(self, a: Optional[int] = None) -> QuantumCircuit:
        """
        Construct the full Shor's algorithm circuit for the given base.

        Parameters
        ----------
        a : Optional[int]
            Base for order finding. If None, uses self.a or chooses randomly.

        Returns
        -------
        QuantumCircuit
            Complete Shor circuit.
        """
        if a is None:
            a = self.choose_coprime()

        n_count = max(8, 2 * math.ceil(math.log2(self.N)))
        n_work = math.ceil(math.log2(self.N))
        total_qubits = n_count + 2 * n_work

        circuit = QuantumCircuit(total_qubits)
        count_qubits = list(range(n_count))
        work_qubits = list(range(n_count, n_count + n_work))

        # Initialize
        for q in count_qubits:
            circuit.h(q)
        circuit.x(work_qubits[0])

        # Modular exponentiation
        mod_exp = ModularExponentiation(a, self.N, n_count)
        circuit.compose(mod_exp.construct_circuit(), inplace=True)

        # Inverse QFT
        from quantumflow.algorithms.qft import InverseQFT
        iqft = InverseQFT(n_count)
        circuit.compose(iqft.construct_circuit(), inplace=True)

        # Measurement
        for q in count_qubits:
            circuit.append(Measurement(), [q])

        return circuit

    def factor(
        self,
        simulator: Optional['StatevectorSimulator'] = None,
        max_attempts: int = 50,
        shots: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run Shor's algorithm to factor N.

        Parameters
        ----------
        simulator : Optional[StatevectorSimulator]
            Quantum simulator.
        max_attempts : int
            Maximum number of attempts with different bases.
        shots : int
            Number of measurement shots.

        Returns
        -------
        Dict[str, Any]
            Result with 'factors', 'N', 'attempts', 'success'.
        """
        N = self.N

        # Check for trivial factors
        if N % 2 == 0:
            self.factors = [2, N // 2]
            return {'factors': self.factors, 'N': N, 'attempts': 0, 'success': True}

        # Check if N is prime
        if self.is_prime(N):
            return {'factors': [N], 'N': N, 'attempts': 0, 'success': True}

        for attempt in range(max_attempts):
            a = self.choose_coprime()
            g = gcd(a, N)

            # Check if we found a factor via GCD
            if 1 < g < N:
                other = N // g
                self.factors = sorted(set([g, other]))
                self._order = None
                return {
                    'factors': self.factors, 'N': N,
                    'attempts': attempt + 1, 'success': True,
                }

            # Quantum order finding
            r = self.order_finding(a, simulator, shots)
            self._order = r

            if r is None:
                continue

            # Extract factors from order
            p, q = self.extract_factors(a, r)
            if p is not None and q is not None:
                self.factors = sorted(set([p, q]))
                return {
                    'factors': self.factors, 'N': N,
                    'order': r, 'base': a,
                    'attempts': attempt + 1, 'success': True,
                }

        return {'factors': [], 'N': N, 'attempts': max_attempts, 'success': False}
