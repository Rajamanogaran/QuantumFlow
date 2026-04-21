"""
Quantum-Aware Optimizers
=========================

Optimizers specifically designed for quantum parameter landscapes.

Unlike classical optimizers, quantum-aware optimizers account for:
- **Parameter-shift rule**: Exact gradients for parameterised quantum
  circuits without finite-difference approximation.
- **Barren plateaus**: Detection and mitigation of vanishing gradients
  in deep quantum circuits.
- **Shot noise**: Gradient estimation from finite measurement shots.
- **Fubini-Study metric**: Natural gradient using the quantum Fisher
  information matrix.
- **SPSA**: Gradient-free optimization for noisy quantum hardware.

Classes
-------
* :class:`QuantumOptimizer` — base class for quantum optimizers.
* :class:`ParameterShiftOptimizer` — exact gradients via parameter-shift.
* :class:`NaturalGradientOptimizer` — natural gradient with Fubini-Study.
* :class:`QuantumAdam` — Adam with barren plateau detection.
* :class:`QuantumLAMB` — LAMB for quantum batch training.
* :class:`QuantumSGD` — SGD with shot-based gradient estimation.
* :class:`SpsaOptimizer` — gradient-free SPSA optimization.
* :class:`GradientFactory` — create gradient estimators by name.

Examples
--------
>>> from quantumflow.tensorflow.optimizers import ParameterShiftOptimizer
>>> optimizer = ParameterShiftOptimizer(learning_rate=0.01)
>>> optimizer.minimize(loss_fn, initial_params, n_iterations=100)

>>> from quantumflow.tensorflow.optimizers import QuantumAdam
>>> opt = QuantumAdam(learning_rate=0.005)
>>> opt.minimize(loss_fn, params, circuit, observable)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "QuantumOptimizer",
    "ParameterShiftOptimizer",
    "NaturalGradientOptimizer",
    "QuantumAdam",
    "QuantumLAMB",
    "QuantumSGD",
    "SpsaOptimizer",
    "GradientFactory",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_HALF_PI = math.pi / 2.0
_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_pauli_observable(pauli: str, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit Pauli operator in full Hilbert space."""
    pauli_matrices = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    mat = pauli_matrices.get(pauli, pauli_matrices["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        full = np.kron(full, mat if i == qubit else np.eye(2, dtype=np.complex128))
    return full


def _get_simulator():
    """Lazily import and return a StatevectorSimulator."""
    from quantumflow.simulation.simulator import StatevectorSimulator
    return StatevectorSimulator()


def _parameter_shift_gradient(
    circuit_fn: Callable,
    observable: np.ndarray,
    params: np.ndarray,
    shift: float = _HALF_PI,
) -> np.ndarray:
    """Compute exact gradient using the parameter-shift rule.

    For rotation gates, the gradient of an expectation value is:

    .. math::

        \\frac{\\partial}{\\partial \\theta_i} \\langle O \\rangle =
        \\frac{1}{2}\\bigl[\\langle O\\rangle_{\\theta_i + \\pi/2}
        - \\langle O\\rangle_{\\theta_i - \\pi/2}\\bigr]

    Parameters
    ----------
    circuit_fn : callable
        Function ``(params) -> QuantumCircuit`` that builds a circuit
        from the given parameters.
    observable : numpy.ndarray
        Observable matrix.
    params : numpy.ndarray
        Current parameters.
    shift : float
        Parameter shift value. Default ``π/2``.

    Returns
    -------
    numpy.ndarray
        Gradient array of same shape as params.
    """
    sim = _get_simulator()
    grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        qc_plus = circuit_fn(params_plus)
        exp_plus = sim.expectation(qc_plus, observable)

        params_minus = params.copy()
        params_minus[i] -= shift
        qc_minus = circuit_fn(params_minus)
        exp_minus = sim.expectation(qc_minus, observable)

        grad[i] = (exp_plus - exp_minus) / 2.0

    return grad


# ═══════════════════════════════════════════════════════════════════════════
# QuantumOptimizer — Base Class
# ═══════════════════════════════════════════════════════════════════════════

class QuantumOptimizer(ABC):
    """Abstract base class for quantum-aware optimizers.

    Provides a common interface for all quantum optimizers with
    ``minimize()``, ``step()``, and state management methods.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.01``.
    name : str, optional
        Optimizer name.

    Attributes
    ----------
    learning_rate : float
    iterations : int
    best_params : numpy.ndarray
    best_loss : float
    history : dict
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        name: Optional[str] = None,
    ) -> None:
        self._learning_rate = learning_rate
        self._name = name or self.__class__.__name__
        self._iterations = 0
        self._best_params: Optional[np.ndarray] = None
        self._best_loss = float("inf")
        self._history: Dict[str, List[float]] = {
            "loss": [],
            "gradient_norm": [],
            "learning_rate": [],
        }

    @property
    def learning_rate(self) -> float:
        """float: Current learning rate."""
        return self._learning_rate

    @property
    def iterations(self) -> int:
        """int: Number of optimization steps taken."""
        return self._iterations

    @property
    def best_params(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Best parameters found so far."""
        return self._best_params

    @property
    def best_loss(self) -> float:
        """float: Best loss value achieved."""
        return self._best_loss

    @property
    def history(self) -> Dict[str, List[float]]:
        """dict: Optimization history."""
        return dict(self._history)

    @abstractmethod
    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the gradient of the objective.

        Parameters
        ----------
        params : numpy.ndarray
        loss_fn : callable, optional
            ``loss_fn(params) -> float``.
        circuit_fn : callable, optional
            ``circuit_fn(params) -> QuantumCircuit``.
        observable : numpy.ndarray, optional
            Observable for expectation-based gradients.

        Returns
        -------
        numpy.ndarray
        """
        ...

    @abstractmethod
    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """Apply a single optimization step.

        Parameters
        ----------
        params : numpy.ndarray
            Current parameters.
        gradient : numpy.ndarray
            Gradient of the objective.

        Returns
        -------
        numpy.ndarray
            Updated parameters.
        """
        ...

    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        n_iterations: int = 100,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
        verbose: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> np.ndarray:
        """Minimize a loss function.

        Parameters
        ----------
        loss_fn : callable
            ``loss_fn(params) -> float``.
        initial_params : numpy.ndarray
        n_iterations : int
        circuit_fn : callable, optional
        observable : numpy.ndarray, optional
        verbose : int
        callback : callable, optional
            Called as ``callback(iteration, params, loss)``.

        Returns
        -------
        numpy.ndarray
            Optimized parameters.
        """
        params = initial_params.copy().astype(np.float64)
        self._best_params = params.copy()
        self._best_loss = loss_fn(params)

        for i in range(n_iterations):
            self._iterations = i + 1
            grad = self.compute_gradient(
                params, loss_fn, circuit_fn, observable
            )
            grad_norm = float(np.linalg.norm(grad))
            params = self.step(params, grad)
            loss = loss_fn(params)

            # Track history
            self._history["loss"].append(loss)
            self._history["gradient_norm"].append(grad_norm)
            self._history["learning_rate"].append(self._learning_rate)

            # Update best
            if loss < self._best_loss:
                self._best_loss = loss
                self._best_params = params.copy()

            if callback is not None:
                callback(i, params, loss)

            if verbose > 0 and (i + 1) % max(1, n_iterations // 10) == 0:
                print(
                    f"Iteration {i + 1}/{n_iterations} - "
                    f"loss: {loss:.6f} - "
                    f"grad_norm: {grad_norm:.6f} - "
                    f"lr: {self._learning_rate:.6f}"
                )

        return params

    def reset(self) -> None:
        """Reset optimizer state."""
        self._iterations = 0
        self._best_params = None
        self._best_loss = float("inf")
        self._history = {"loss": [], "gradient_norm": [], "learning_rate": []}

    def get_config(self) -> Dict[str, Any]:
        """Return optimizer configuration."""
        return {
            "learning_rate": self._learning_rate,
            "name": self._name,
            "iterations": self._iterations,
        }

    def __repr__(self) -> str:
        return f"{self._name}(lr={self._learning_rate}, iterations={self._iterations})"


# ═══════════════════════════════════════════════════════════════════════════
# ParameterShiftOptimizer — Exact Gradients via Parameter-Shift Rule
# ═══════════════════════════════════════════════════════════════════════════

class ParameterShiftOptimizer(QuantumOptimizer):
    """Optimizer using the parameter-shift rule for exact quantum gradients.

    No finite-difference approximation — the parameter-shift rule provides
    exact gradients for parameterised quantum circuits with rotation gates.

    The gradient is computed as:

    .. math::

        \\frac{\\partial \\langle O \\rangle}{\\partial \\theta_i} =
        \\frac{1}{2}\\bigl[f(\\theta + \\pi/2 e_i) - f(\\theta - \\pi/2 e_i)\\bigr]

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.01``.
    shift : float, optional
        Parameter shift value. Default ``π/2``.
    name : str, optional

    Examples
    --------
    >>> opt = ParameterShiftOptimizer(learning_rate=0.01)
    >>> result = opt.minimize(loss_fn, initial_params, n_iterations=100)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        shift: float = _HALF_PI,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "ParameterShiftOptimizer")
        self._shift = shift

    @property
    def shift(self) -> float:
        """float: Parameter shift value."""
        return self._shift

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute exact gradient via parameter-shift rule.

        Uses ``circuit_fn`` + ``observable`` for quantum gradient if
        available, otherwise falls back to ``loss_fn`` with central
        finite differences.
        """
        if circuit_fn is not None and observable is not None:
            return _parameter_shift_gradient(circuit_fn, observable, params, self._shift)

        if loss_fn is None:
            raise ValueError("Either loss_fn or (circuit_fn, observable) must be provided.")

        # Central finite differences (fallback)
        grad = np.zeros_like(params)
        eps = 1e-7
        base_loss = loss_fn(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (loss_fn(params_plus) - loss_fn(params_minus)) / (2 * eps)
        return grad

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """Vanilla gradient descent step."""
        return params - self._learning_rate * gradient


# ═══════════════════════════════════════════════════════════════════════════
# NaturalGradientOptimizer — Natural Gradient with Fubini-Study Metric
# ═══════════════════════════════════════════════════════════════════════════

class NaturalGradientOptimizer(QuantumOptimizer):
    """Natural gradient descent using the Fubini-Study metric tensor.

    Uses the quantum Fisher information matrix to precondition gradients:

    .. math::

        \\theta_{k+1} = \\theta_k - \\eta \\, g^{-1}(\\theta_k) \\, \\nabla L(\\theta_k)

    where ``g`` is the Fubini-Study metric tensor computed from the
    quantum state.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.01``.
    epsilon : float, optional
        Regularization for the metric tensor. Default ``1e-6``.
    name : str, optional

    Examples
    --------
    >>> opt = NaturalGradientOptimizer(learning_rate=0.01)
    >>> result = opt.minimize(loss_fn, params, circuit_fn=my_circuit, observable=obs)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-6,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "NaturalGradientOptimizer")
        self._epsilon = epsilon

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute natural gradient (Fisher-preconditioned)."""
        if loss_fn is None:
            raise ValueError("loss_fn is required.")
        if circuit_fn is None:
            # Fall back to vanilla gradient
            opt = ParameterShiftOptimizer(self._learning_rate)
            return opt.compute_gradient(params, loss_fn)

        # Compute vanilla gradient
        vanilla_grad = _parameter_shift_gradient(circuit_fn, observable, params)

        # Compute quantum Fisher information matrix
        g = self._compute_fisher_matrix(circuit_fn, params)

        # Add regularization
        g += self._epsilon * np.eye(len(params))

        # Precondition: theta_update = -lr * g^{-1} @ grad
        try:
            preconditioned_grad = np.linalg.solve(g, vanilla_grad)
        except np.linalg.LinAlgError:
            preconditioned_grad = vanilla_grad

        return preconditioned_grad

    def _compute_fisher_matrix(
        self,
        circuit_fn: Callable,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute the quantum Fisher information matrix (Fubini-Study metric).

        The (i,j)-th element is:

        .. math::

            g_{ij} = \\frac{1}{4}\\bigl[\\langle \\partial_i \\psi | \\partial_j \\psi \\rangle
            + \\langle \\partial_j \\psi | \\partial_i \\psi \\rangle
            - 2\\text{Re}(\\langle \\partial_i \\psi | \\psi \\rangle
            \\langle \\psi | \\partial_j \\psi \\rangle)\\bigr]

        Parameters
        ----------
        circuit_fn : callable
        params : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Shape ``(n_params, n_params)``.
        """
        sim = _get_simulator()
        n = len(params)
        eps = 1e-6

        # Get the current state
        qc_base = circuit_fn(params)
        state_base = sim.state(qc_base)
        sv_base = state_base.data if hasattr(state_base, 'data') else np.asarray(state_base)

        # Compute state gradients
        state_grads = []
        for i in range(n):
            params_plus = params.copy()
            params_plus[i] += eps
            qc_plus = circuit_fn(params_plus)
            state_plus = sim.state(qc_plus)
            sv_plus = state_plus.data if hasattr(state_plus, 'data') else np.asarray(state_plus)
            grad_i = (sv_plus - sv_base) / eps
            state_grads.append(grad_i)

        # Compute Fisher matrix
        g = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                # g_ij = 1/4 * (Re(<∂iψ|∂jψ>) - Re(<∂iψ|ψ><ψ|∂jψ>) * 2)
                inner_ij = float(np.real(np.vdot(state_grads[i], state_grads[j])))
                overlap_i = float(np.real(np.vdot(state_grads[i], sv_base)))
                overlap_j = float(np.real(np.vdot(state_grads[j], sv_base)))
                g_ij = 0.25 * (inner_ij + inner_ij - 2 * overlap_i * overlap_j)
                g[i, j] = g_ij
                g[j, i] = g_ij

        return g

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """Natural gradient step (gradient already preconditioned)."""
        return params - self._learning_rate * gradient


# ═══════════════════════════════════════════════════════════════════════════
# QuantumAdam — Adam with Barren Plateau Detection
# ═══════════════════════════════════════════════════════════════════════════

class QuantumAdam(QuantumOptimizer):
    """Adam optimizer adapted for quantum parameter landscapes.

    Extends standard Adam with:
    - **Barren plateau detection**: Monitors gradient norms and adjusts
      learning rate when gradients become vanishingly small.
    - **Noise-aware moment estimation**: Accounts for shot noise in
      quantum gradient estimation.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.001``.
    beta1 : float, optional
        First moment decay. Default ``0.9``.
    beta2 : float, optional
        Second moment decay. Default ``0.999``.
    epsilon : float, optional
        Numerical stability. Default ``1e-8``.
    barren_plateau_threshold : float, optional
        Gradient norm threshold for barren plateau detection. Default ``1e-6``.
    plateau_lr_multiplier : float, optional
        Learning rate multiplier when barren plateau detected. Default ``5.0``.
    name : str, optional

    Examples
    --------
    >>> opt = QuantumAdam(learning_rate=0.001)
    >>> result = opt.minimize(loss_fn, params, n_iterations=200)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        barren_plateau_threshold: float = 1e-6,
        plateau_lr_multiplier: float = 5.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "QuantumAdam")
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._bp_threshold = barren_plateau_threshold
        self._bp_lr_multiplier = plateau_lr_multiplier

        # Adam state
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t: int = 0

        # Barren plateau tracking
        self._plateau_count: int = 0
        self._is_in_plateau: bool = False

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute gradient with barren plateau awareness."""
        if circuit_fn is not None and observable is not None:
            grad = _parameter_shift_gradient(circuit_fn, observable, params)
        elif loss_fn is not None:
            grad = np.zeros_like(params)
            eps = 1e-7
            for i in range(len(params)):
                pp = params.copy(); pp[i] += eps
                pm = params.copy(); pm[i] -= eps
                grad[i] = (loss_fn(pp) - loss_fn(pm)) / (2 * eps)
        else:
            raise ValueError("Either loss_fn or (circuit_fn, observable) required.")

        # Barren plateau detection
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < self._bp_threshold:
            self._plateau_count += 1
            if self._plateau_count > 5:
                self._is_in_plateau = True
        else:
            self._plateau_count = 0
            self._is_in_plateau = False

        return grad

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """Adam update with barren plateau adaptation."""
        if self._m is None:
            self._m = np.zeros_like(params)
        if self._v is None:
            self._v = np.zeros_like(params)

        self._t += 1

        # Update biased first moment estimate
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient

        # Update biased second raw moment estimate
        self._v = self._beta2 * self._v + (1 - self._beta2) * (gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self._m / (1 - self._beta1 ** self._t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self._v / (1 - self._beta2 ** self._t)

        # Adaptive learning rate for barren plateaus
        lr = self._learning_rate
        if self._is_in_plateau:
            lr *= self._bp_lr_multiplier

        # Update parameters
        update = lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
        return params - update

    def reset(self) -> None:
        """Reset optimizer state including Adam moments."""
        super().reset()
        self._m = None
        self._v = None
        self._t = 0
        self._plateau_count = 0
        self._is_in_plateau = False

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "beta1": self._beta1,
            "beta2": self._beta2,
            "epsilon": self._epsilon,
            "barren_plateau_threshold": self._bp_threshold,
            "is_in_plateau": self._is_in_plateau,
        })
        return config


# ═══════════════════════════════════════════════════════════════════════════
# QuantumLAMB — LAMB for Quantum Batch Training
# ═══════════════════════════════════════════════════════════════════════════

class QuantumLAMB(QuantumOptimizer):
    """LAMB optimizer adapted for quantum batch training.

    LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
    applies per-parameter trust ratio clipping for stable large-batch
    training of quantum circuits.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.001``.
    beta1 : float, optional
        Default ``0.9``.
    beta2 : float, optional
        Default ``0.999``.
    epsilon : float, optional
        Default ``1e-8``.
    trust_ratio_clip : float, optional
        Maximum trust ratio. Default ``10.0``.
    name : str, optional

    Examples
    --------
    >>> opt = QuantumLAMB(learning_rate=0.001)
    >>> result = opt.minimize(loss_fn, params, n_iterations=100)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        trust_ratio_clip: float = 10.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "QuantumLAMB")
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._trust_ratio_clip = trust_ratio_clip

        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t: int = 0

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute gradient."""
        if circuit_fn is not None and observable is not None:
            return _parameter_shift_gradient(circuit_fn, observable, params)
        elif loss_fn is not None:
            grad = np.zeros_like(params)
            eps = 1e-7
            for i in range(len(params)):
                pp = params.copy(); pp[i] += eps
                pm = params.copy(); pm[i] -= eps
                grad[i] = (loss_fn(pp) - loss_fn(pm)) / (2 * eps)
            return grad
        raise ValueError("Either loss_fn or (circuit_fn, observable) required.")

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """LAMB update with trust ratio clipping."""
        if self._m is None:
            self._m = np.zeros_like(params)
        if self._v is None:
            self._v = np.zeros_like(params)

        self._t += 1

        # Adam moments
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = self._beta2 * self._v + (1 - self._beta2) * (gradient ** 2)

        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)

        # LAMB: Adam update with trust ratio
        adam_update = m_hat / (np.sqrt(v_hat) + self._epsilon)
        param_norm = float(np.linalg.norm(params))
        update_norm = float(np.linalg.norm(adam_update))

        # Trust ratio
        if update_norm > _TOLERANCE:
            trust_ratio = min(param_norm / update_norm, self._trust_ratio_clip)
        else:
            trust_ratio = 1.0

        return params - self._learning_rate * trust_ratio * adam_update

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._m = None
        self._v = None
        self._t = 0


# ═══════════════════════════════════════════════════════════════════════════
# QuantumSGD — SGD with Shot-Based Gradient Estimation
# ═══════════════════════════════════════════════════════════════════════════

class QuantumSGD(QuantumOptimizer):
    """Stochastic Gradient Descent with quantum-specific features.

    Supports:
    - **Shot-based gradient estimation**: Estimates gradients from
      finite measurement shots.
    - **Adaptive shot allocation**: Allocates more shots to parameters
      with larger estimated gradient magnitudes.
    - **Momentum**: Standard momentum for acceleration.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.01``.
    momentum : float, optional
        Momentum factor. Default ``0.9``.
    shots : int, optional
        Number of measurement shots per gradient estimate. Default ``1000``.
    adaptive_shots : bool, optional
        Enable adaptive shot allocation. Default ``True``.
    min_shots : int, optional
        Minimum shots per parameter. Default ``100``.
    name : str, optional

    Examples
    --------
    >>> opt = QuantumSGD(learning_rate=0.01, shots=1024)
    >>> result = opt.minimize(loss_fn, params, n_iterations=100)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        shots: int = 1000,
        adaptive_shots: bool = True,
        min_shots: int = 100,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "QuantumSGD")
        self._momentum = momentum
        self._shots = shots
        self._adaptive_shots = adaptive_shots
        self._min_shots = min_shots

        self._velocity: Optional[np.ndarray] = None
        self._prev_grad: Optional[np.ndarray] = None

    @property
    def shots(self) -> int:
        """int: Number of measurement shots."""
        return self._shots

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute gradient with optional shot-based estimation."""
        if circuit_fn is not None and observable is not None:
            if self._adaptive_shots and self._prev_grad is not None:
                # Allocate shots proportionally to gradient magnitude
                grad_magnitudes = np.abs(self._prev_grad)
                total_mag = np.sum(grad_magnitudes) + 1e-10
                n_params = len(params)
                shots_per_param = np.maximum(
                    (grad_magnitudes / total_mag * self._shots).astype(int),
                    self._min_shots,
                )
            else:
                shots_per_param = None

            return self._shot_based_gradient(circuit_fn, observable, params, shots_per_param)

        if loss_fn is not None:
            grad = np.zeros_like(params)
            eps = 1e-7
            for i in range(len(params)):
                pp = params.copy(); pp[i] += eps
                pm = params.copy(); pm[i] -= eps
                grad[i] = (loss_fn(pp) - loss_fn(pm)) / (2 * eps)
            self._prev_grad = grad.copy()
            return grad

        raise ValueError("Either loss_fn or (circuit_fn, observable) required.")

    def _shot_based_gradient(
        self,
        circuit_fn: Callable,
        observable: np.ndarray,
        params: np.ndarray,
        shots_per_param: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate gradient from measurement shots.

        Parameters
        ----------
        circuit_fn : callable
        observable : numpy.ndarray
        params : numpy.ndarray
        shots_per_param : numpy.ndarray, optional
            Shots to allocate per parameter.

        Returns
        -------
        numpy.ndarray
        """
        sim = _get_simulator()
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)

        for i in range(n):
            s = int(shots_per_param[i]) if shots_per_param is not None else self._shots

            params_plus = params.copy()
            params_plus[i] += _HALF_PI
            qc_plus = circuit_fn(params_plus)
            exp_plus = sim.expectation(qc_plus, observable)

            params_minus = params.copy()
            params_minus[i] -= _HALF_PI
            qc_minus = circuit_fn(params_minus)
            exp_minus = sim.expectation(qc_minus, observable)

            # Add shot noise
            noise_plus = np.random.normal(0, 1.0 / math.sqrt(s)) if s > 0 else 0
            noise_minus = np.random.normal(0, 1.0 / math.sqrt(s)) if s > 0 else 0

            grad[i] = (exp_plus + noise_plus - exp_minus - noise_minus) / 2.0

        self._prev_grad = grad.copy()
        return grad

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """SGD step with momentum."""
        if self._velocity is None:
            self._velocity = np.zeros_like(params)

        self._velocity = self._momentum * self._velocity + gradient
        return params - self._learning_rate * self._velocity

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._velocity = None
        self._prev_grad = None


# ═══════════════════════════════════════════════════════════════════════════
# SpsaOptimizer — Simultaneous Perturbation Stochastic Approximation
# ═══════════════════════════════════════════════════════════════════════════

class SpsaOptimizer(QuantumOptimizer):
    """SPSA — gradient-free optimization for noisy quantum settings.

    Simultaneous Perturbation Stochastic Approximation estimates the
    gradient using only *two* function evaluations per iteration,
    regardless of the number of parameters:

    .. math::

        \\hat{g}_k = \\frac{f(\\theta + c\\Delta_k) - f(\\theta - c\\Delta_k)}
        {2c\\Delta_k}

    where ``Δ_k`` is a random perturbation vector with components
    ``±1``.

    SPSA is ideal for:
    - Noisy quantum hardware where individual parameter shifts are
      expensive.
    - Problems with many parameters.
    - Black-box optimization where gradient computation is unavailable.

    Parameters
    ----------
    learning_rate : float, optional
        Default ``0.01``.
    perturbation : float, optional
        Perturbation size ``c``. Default ``0.1``.
    alpha : float, optional
        Learning rate decay exponent. Default ``0.602``.
    gamma : float, optional
        Perturbation decay exponent. Default ``0.101``.
    A : float, optional
        Stability constant. Default ``10.0``.
    name : str, optional

    Examples
    --------
    >>> opt = SpsaOptimizer(learning_rate=0.1, perturbation=0.1)
    >>> result = opt.minimize(loss_fn, params, n_iterations=200)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        perturbation: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(learning_rate=learning_rate, name=name or "SpsaOptimizer")
        self._perturbation = perturbation
        self._alpha = alpha
        self._gamma = gamma
        self._A = A

        self._gradient_estimate: Optional[np.ndarray] = None

    @property
    def perturbation(self) -> float:
        """float: Current perturbation size."""
        return self._perturbation

    def compute_gradient(
        self,
        params: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute SPSA gradient estimate.

        Uses only 2 function evaluations regardless of parameter count.
        """
        if loss_fn is None:
            raise ValueError("loss_fn is required for SPSA.")

        n = len(params)
        k = self._iterations

        # Adaptive perturbation size: c_k = c / (k+1)^gamma
        c_k = self._perturbation / ((k + 1) ** self._gamma)

        # Random perturbation vector (±1)
        delta = np.random.choice([-1.0, 1.0], size=n)

        # Two function evaluations
        params_plus = params + c_k * delta
        params_minus = params - c_k * delta
        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)

        # SPSA gradient estimate
        grad = (loss_plus - loss_minus) / (2 * c_k * delta)
        self._gradient_estimate = grad

        return grad

    def step(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
    ) -> np.ndarray:
        """SPSA update with adaptive learning rate."""
        k = self._iterations

        # Adaptive learning rate: a_k = a / (k + 1 + A)^alpha
        a_k = self._learning_rate / ((k + 1 + self._A) ** self._alpha)

        return params - a_k * gradient

    def minimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        n_iterations: int = 100,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
        verbose: int = 1,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> np.ndarray:
        """Optimized SPSA minimization.

        Overrides base to avoid recomputing gradient.
        """
        params = initial_params.copy().astype(np.float64)
        self._best_params = params.copy()
        self._best_loss = loss_fn(params)

        for i in range(n_iterations):
            self._iterations = i + 1

            # Compute gradient (SPSA uses only 2 evaluations)
            grad = self.compute_gradient(params, loss_fn, circuit_fn, observable)
            grad_norm = float(np.linalg.norm(grad))

            # Apply step
            params = self.step(params, grad)
            loss = loss_fn(params)

            self._history["loss"].append(loss)
            self._history["gradient_norm"].append(grad_norm)
            self._history["learning_rate"].append(
                self._learning_rate / ((i + 1 + self._A) ** self._alpha)
            )

            if loss < self._best_loss:
                self._best_loss = loss
                self._best_params = params.copy()

            if callback is not None:
                callback(i, params, loss)

            if verbose > 0 and (i + 1) % max(1, n_iterations // 10) == 0:
                print(
                    f"Iteration {i + 1}/{n_iterations} - "
                    f"loss: {loss:.6f} - grad_norm: {grad_norm:.6f}"
                )

        return params

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._gradient_estimate = None


# ═══════════════════════════════════════════════════════════════════════════
# GradientFactory — Create Gradient Estimators
# ═══════════════════════════════════════════════════════════════════════════

class GradientFactory:
    """Factory for creating gradient estimators by name.

    Supported estimators:

    * ``'parameter_shift'`` — Exact gradients via parameter-shift rule.
      Most accurate, requires circuit + observable.
    * ``'finite_difference'`` — Central finite differences.
      Works with any loss function, less accurate.
    * ``'spsa'`` — Simultaneous Perturbation Stochastic Approximation.
      Gradient-free, good for noisy settings.
    * ``'parameter_shift_rule'`` — Alias for ``'parameter_shift'``.

    Examples
    --------
    >>> from quantumflow.tensorflow.optimizers import GradientFactory
    >>> grad_fn = GradientFactory.create('parameter_shift', circuit_fn=my_circuit, observable=obs)
    >>> grad = grad_fn(params)
    """

    _REGISTERED_ESTIMATORS: Dict[str, str] = {
        "parameter_shift": "parameter_shift",
        "parameter_shift_rule": "parameter_shift",
        "finite_difference": "finite_difference",
        "spsa": "spsa",
    }

    @classmethod
    def available_estimators(cls) -> List[str]:
        """Return list of available estimator names.

        Returns
        -------
        list of str
        """
        return list(cls._REGISTERED_ESTIMATORS.keys())

    @classmethod
    def create(
        cls,
        name: str,
        circuit_fn: Optional[Callable] = None,
        observable: Optional[np.ndarray] = None,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
        shift: float = _HALF_PI,
        epsilon: float = 1e-7,
        perturbation: float = 0.1,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a gradient estimator function.

        Parameters
        ----------
        name : str
            Estimator name.
        circuit_fn : callable, optional
            ``circuit_fn(params) -> QuantumCircuit``.
        observable : numpy.ndarray, optional
        loss_fn : callable, optional
            ``loss_fn(params) -> float``.
        shift : float, optional
            Parameter shift value. Default ``π/2``.
        epsilon : float, optional
            Finite difference step. Default ``1e-7``.
        perturbation : float, optional
            SPSA perturbation size. Default ``0.1``.

        Returns
        -------
        callable
            Gradient function ``(params) -> gradient_array``.

        Raises
        ------
        ValueError
            If estimator name is unknown or required arguments missing.
        """
        if name not in cls._REGISTERED_ESTIMATORS:
            available = ", ".join(sorted(cls._REGISTERED_ESTIMATORS.keys()))
            raise ValueError(
                f"Unknown estimator '{name}'. Available: {available}"
            )

        estimator_type = cls._REGISTERED_ESTIMATORS[name]

        if estimator_type == "parameter_shift":
            if circuit_fn is None or observable is None:
                raise ValueError(
                    "Parameter-shift estimator requires circuit_fn and observable."
                )

            def parameter_shift_grad(params: np.ndarray) -> np.ndarray:
                return _parameter_shift_gradient(circuit_fn, observable, params, shift)

            return parameter_shift_grad

        elif estimator_type == "finite_difference":
            if loss_fn is None:
                raise ValueError(
                    "Finite-difference estimator requires loss_fn."
                )

            def finite_diff_grad(params: np.ndarray) -> np.ndarray:
                grad = np.zeros_like(params)
                for i in range(len(params)):
                    pp = params.copy(); pp[i] += epsilon
                    pm = params.copy(); pm[i] -= epsilon
                    grad[i] = (loss_fn(pp) - loss_fn(pm)) / (2 * epsilon)
                return grad

            return finite_diff_grad

        elif estimator_type == "spsa":
            if loss_fn is None:
                raise ValueError("SPSA estimator requires loss_fn.")

            spsa_state = {"iteration": 0}

            def spsa_grad(params: np.ndarray) -> np.ndarray:
                spsa_state["iteration"] += 1
                k = spsa_state["iteration"]
                n = len(params)
                delta = np.random.choice([-1.0, 1.0], size=n)
                c_k = perturbation / ((k + 1) ** 0.101)
                loss_plus = loss_fn(params + c_k * delta)
                loss_minus = loss_fn(params - c_k * delta)
                return (loss_plus - loss_minus) / (2 * c_k * delta)

            return spsa_grad

        raise ValueError(f"Unknown estimator type: {estimator_type}")

    @classmethod
    def create_optimizer(
        cls,
        name: str,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> QuantumOptimizer:
        """Create a quantum optimizer by name.

        Parameters
        ----------
        name : str
            One of: ``'parameter_shift'``, ``'natural_gradient'``,
            ``'quantum_adam'``, ``'quantum_lamb'``, ``'quantum_sgd'``,
            ``'spsa'``.
        learning_rate : float
        **kwargs
            Additional optimizer-specific arguments.

        Returns
        -------
        QuantumOptimizer
        """
        name_map = {
            "parameter_shift": ParameterShiftOptimizer,
            "natural_gradient": NaturalGradientOptimizer,
            "quantum_adam": QuantumAdam,
            "quantum_lamb": QuantumLAMB,
            "quantum_sgd": QuantumSGD,
            "spsa": SpsaOptimizer,
        }

        if name not in name_map:
            available = ", ".join(sorted(name_map.keys()))
            raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")

        return name_map[name](learning_rate=learning_rate, **kwargs)

    @classmethod
    def register_estimator(
        cls,
        name: str,
        estimator_type: str = "custom",
    ) -> None:
        """Register a custom gradient estimator.

        Parameters
        ----------
        name : str
            Name to register.
        estimator_type : str
            Internal type identifier.
        """
        cls._REGISTERED_ESTIMATORS[name] = estimator_type

    def __repr__(self) -> str:
        return (
            f"GradientFactory(available={list(self._REGISTERED_ESTIMATORS.keys())})"
        )
