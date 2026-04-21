"""
Mathematical Utilities for Quantum Computing
=============================================
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union


def kron(*matrices: np.ndarray) -> np.ndarray:
    """
    Compute the Kronecker product of multiple matrices.

    Parameters
    ----------
    *matrices : np.ndarray
        Matrices to tensor product together.

    Returns
    -------
    np.ndarray
        Kronecker product of all input matrices.

    Examples
    --------
    >>> kron(np.eye(2), np.array([[0,1],[1,0]]))
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def tensor_product(*states: np.ndarray) -> np.ndarray:
    """
    Compute the tensor product of quantum state vectors.

    Parameters
    ----------
    *states : np.ndarray
        State vectors to tensor together.

    Returns
    -------
    np.ndarray
        Tensor product state.

    Examples
    --------
    >>> tensor_product(np.array([1, 0]), np.array([0, 1]))
    array([0., 1., 0., 0.])
    """
    result = states[0]
    for s in states[1:]:
        result = np.kron(result, s)
    return result


def partial_trace(rho: np.ndarray, qubits_to_keep: List[int], n_qubits: int) -> np.ndarray:
    """
    Compute the partial trace of a density matrix.

    Traces out qubits not in qubits_to_keep.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of shape (2^n, 2^n).
    qubits_to_keep : List[int]
        Qubits to keep (trace out the rest).
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Reduced density matrix.

    Examples
    --------
    >>> # Trace out qubit 1 from a 2-qubit state
    >>> rho = np.eye(4) / 4  # Maximally mixed state
    >>> reduced = partial_trace(rho, [0], 2)
    >>> assert np.allclose(reduced, np.eye(2) / 2)
    """
    d = 2 ** n_qubits
    rho = rho.reshape([2] * 2 * n_qubits)

    qubits_to_trace = [q for q in range(n_qubits) if q not in qubits_to_keep]

    for q in sorted(qubits_to_trace, reverse=True):
        # Trace over qubit q: sum over axes (q, q + n_qubits)
        rho = np.trace(rho, axis1=q, axis2=q + n_qubits)
        # Reshape back
        remaining = len([x for x in qubits_to_trace if x < q])
        new_shape = [2] * 2 * (n_qubits - 1 - remaining)
        rho = rho.reshape(new_shape)

    d_keep = 2 ** len(qubits_to_keep)
    return rho.reshape(d_keep, d_keep)


def density_from_statevector(psi: np.ndarray) -> np.ndarray:
    """
    Convert a state vector to a density matrix.

    Parameters
    ----------
    psi : np.ndarray
        Pure state vector.

    Returns
    -------
    np.ndarray
        Density matrix |psi><psi|.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    return np.outer(psi, psi.conj())


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the Uhlmann fidelity between two quantum states.

    F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2

    For pure states, reduces to |<psi|phi>|^2.

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices or state vectors.

    Returns
    -------
    float
        Fidelity in [0, 1].

    Examples
    --------
    >>> fidelity(np.array([1, 0]), np.array([1, 0]))
    1.0
    >>> fidelity(np.array([1, 0]), np.array([0, 1]))
    0.0
    """
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)

    # Handle pure states (1D vectors)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if sigma.ndim == 1:
        sigma = np.outer(sigma, sigma.conj())

    # For pure-pure case
    if rho.ndim == 2 and sigma.ndim == 2:
        eigvals = np.linalg.eigvalsh(rho)
        if np.max(np.abs(eigvals - 1.0)) < 1e-10:
            # rho is pure
            idx = np.argmax(eigvals)
            psi = np.linalg.eigh(rho)[1][:, idx]
            return float(np.abs(np.conj(psi) @ sigma @ psi) ** 2)
        eigvals = np.linalg.eigvalsh(sigma)
        if np.max(np.abs(eigvals - 1.0)) < 1e-10:
            idx = np.argmax(eigvals)
            phi = np.linalg.eigh(sigma)[1][:, idx]
            return float(np.abs(np.conj(phi) @ rho @ phi) ** 2)

    # General case: use eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)  # Numerical stability
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    product = sqrt_rho @ sigma @ sqrt_rho
    eigvals_prod = np.linalg.eigvalsh(product)
    eigvals_prod = np.maximum(eigvals_prod, 0)
    return float(np.sum(np.sqrt(eigvals_prod)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the trace distance between two quantum states.

    T(rho, sigma) = 0.5 * Tr|rho - sigma|

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices or state vectors.

    Returns
    -------
    float
        Trace distance in [0, 1].
    """
    rho = np.asarray(rho, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=np.complex128)

    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if sigma.ndim == 1:
        sigma = np.outer(sigma, sigma.conj())

    diff = rho - sigma
    return float(0.5 * np.linalg.norm(diff, ord='nuc'))


def purity(rho: np.ndarray) -> float:
    """
    Compute the purity of a quantum state.

    P(rho) = Tr(rho^2)

    Pure states have purity 1, maximally mixed states have purity 1/d.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix or state vector.

    Returns
    -------
    float
        Purity in [1/d, 1].

    Examples
    --------
    >>> purity(np.array([1, 0]))  # Pure state
    1.0
    >>> purity(np.eye(2) / 2)  # Maximally mixed
    0.5
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim == 1:
        return 1.0
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, base: float = 2) -> float:
    """
    Compute the von Neumann entropy of a quantum state.

    S(rho) = -Tr(rho * log(rho))

    For pure states S = 0, for maximally mixed S = log(d).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix or state vector.
    base : float
        Logarithm base (2 for bits, e for nats).

    Returns
    -------
    float
        Entropy >= 0.

    Examples
    --------
    >>> von_neumann_entropy(np.array([1, 0]))  # Pure state
    0.0
    >>> von_neumann_entropy(np.eye(2) / 2)  # Maximally mixed
    1.0
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim == 1:
        return 0.0

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # Remove near-zero eigenvalues

    entropy = -np.sum(eigvals * np.log(eigvals))
    if base == 2:
        entropy /= np.log(2)
    elif base == 10:
        entropy /= np.log(10)

    return float(entropy)


def expectation_value(
    state: np.ndarray,
    operator: np.ndarray,
) -> complex:
    """
    Compute the expectation value of an operator.

    <psi|O|psi> for pure states, or Tr(rho * O) for mixed states.

    Parameters
    ----------
    state : np.ndarray
        State vector or density matrix.
    operator : np.ndarray
        Observable operator (Hermitian matrix).

    Returns
    -------
    complex
        Expectation value.
    """
    state = np.asarray(state, dtype=np.complex128)
    operator = np.asarray(operator, dtype=np.complex128)

    if state.ndim == 1:
        return complex(np.conj(state) @ operator @ state)
    else:
        return complex(np.trace(state @ operator))


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the anticommutator {A, B} = AB + BA."""
    return A @ B + B @ A


def is_hermitian(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=atol)


def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a matrix is unitary."""
    n = matrix.shape[0]
    return np.allclose(matrix @ matrix.conj().T, np.eye(n), atol=atol)


def is_positive_semidefinite(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check if a matrix is positive semidefinite."""
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -atol)


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize a quantum state vector."""
    state = np.asarray(state, dtype=np.complex128)
    norm = np.linalg.norm(state)
    if norm > 0:
        return state / norm
    return state


def random_unitary(n: int) -> np.ndarray:
    """Generate a random n x n unitary matrix (Haar measure)."""
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R) / np.abs(np.diag(R))
    return Q @ D


def random_density_matrix(n: int, rank: Optional[int] = None) -> np.ndarray:
    """Generate a random density matrix."""
    if rank is None:
        rank = n
    U = random_unitary(n)
    eigenvalues = np.random.dirichlet(np.ones(rank))
    padded_eigenvalues = np.zeros(n)
    padded_eigenvalues[:rank] = eigenvalues
    return U @ np.diag(padded_eigenvalues) @ U.conj().T


def pauli_matrices() -> Dict[str, np.ndarray]:
    """Return the Pauli matrices."""
    return {
        'I': np.eye(2, dtype=np.complex128),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }


def state_to_bloch(state: np.ndarray) -> np.ndarray:
    """Convert a single-qubit state to Bloch sphere coordinates."""
    state = normalize_state(state)
    alpha, beta = state
    x = 2.0 * np.real(alpha * np.conj(beta))
    y = 2.0 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    return np.array([x, y, z])


def bloch_to_state(bloch: np.ndarray) -> np.ndarray:
    """Convert Bloch sphere coordinates to a quantum state."""
    x, y, z = bloch
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([alpha, beta], dtype=np.complex128)
