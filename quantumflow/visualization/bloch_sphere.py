"""
Bloch Sphere Visualization
===========================

Interactive Bloch sphere visualization for single-qubit states.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any


class BlochSphere:
    """
    Bloch sphere visualization for quantum states.

    Represents a single-qubit state |ψ> = α|0> + β|1> as a point
    on the Bloch sphere with coordinates:
        x = 2*Re(α*β*)
        y = 2*Im(α*β*)
        z = |α|^2 - |β|^2

    Parameters
    ----------
    figsize : Tuple[float, float]
        Figure size in inches.
    title : Optional[str]
        Plot title.

    Examples
    --------
    >>> bloch = BlochSphere()
    >>> bloch.add_state(np.array([1, 0]))  # |0> at north pole
    >>> bloch.add_state(np.array([1, 1]) / np.sqrt(2))  # |+> on x-axis
    >>> bloch.show()

    >>> # Multiple states with labels
    >>> bloch = BlochSphere()
    >>> states = {
    ...     '|0⟩': np.array([1, 0]),
    ...     '|1⟩': np.array([0, 1]),
    ...     '|+⟩': np.array([1, 1]) / np.sqrt(2),
    ...     '|-⟩': np.array([1, -1]) / np.sqrt(2),
    ...     '|+i⟩': np.array([1, 1j]) / np.sqrt(2),
    ... }
    >>> for label, state in states.items():
    ...     bloch.add_state(state, label=label)
    >>> bloch.show()
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (8, 8),
        title: Optional[str] = None,
    ) -> None:
        self.figsize = figsize
        self.title = title
        self._states: List[Dict[str, Any]] = []
        self._arrows: List[Dict[str, Any]] = []
        self._trails: List[List[np.ndarray]] = []

    @staticmethod
    def state_to_bloch(state: np.ndarray) -> np.ndarray:
        """
        Convert a quantum state vector to Bloch sphere coordinates.

        Parameters
        ----------
        state : np.ndarray
            Single-qubit state vector [α, β].

        Returns
        -------
        np.ndarray
            Bloch vector [x, y, z].

        Raises
        ------
        ValueError
            If state is not normalized or not 2-dimensional.
        """
        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (2,):
            raise ValueError(f"State must be 2-dimensional, got shape {state.shape}")

        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-6):
            state = state / norm

        alpha, beta = state
        x = 2.0 * np.real(alpha * np.conj(beta))
        y = 2.0 * np.imag(alpha * np.conj(beta))
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
        return np.array([x, y, z])

    @staticmethod
    def bloch_to_state(bloch_vector: np.ndarray) -> np.ndarray:
        """
        Convert Bloch sphere coordinates to a quantum state.

        Parameters
        ----------
        bloch_vector : np.ndarray
            Bloch vector [x, y, z].

        Returns
        -------
        np.ndarray
            State vector [α, β].
        """
        x, y, z = bloch_vector
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)

        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return np.array([alpha, beta], dtype=np.complex128)

    @staticmethod
    def bloch_angles(state: np.ndarray) -> Tuple[float, float]:
        """
        Get the Bloch sphere angles (theta, phi) for a state.

        Parameters
        ----------
        state : np.ndarray
            Single-qubit state.

        Returns
        -------
        Tuple[float, float]
            (theta, phi) in radians.
        """
        bloch = BlochSphere.state_to_bloch(state)
        theta = np.arccos(np.clip(bloch[2], -1, 1))
        phi = np.arctan2(bloch[1], bloch[0])
        return theta, phi

    def add_state(
        self,
        state: np.ndarray,
        label: Optional[str] = None,
        color: str = 'red',
        point_size: int = 100,
        alpha: float = 1.0,
    ) -> None:
        """
        Add a quantum state to the Bloch sphere.

        Parameters
        ----------
        state : np.ndarray
            Single-qubit state vector.
        label : Optional[str]
            Label for the state.
        color : str
            Color for the point.
        point_size : int
            Size of the marker.
        alpha : float
            Transparency.
        """
        bloch = self.state_to_bloch(state)
        self._states.append({
            'bloch': bloch,
            'state': state,
            'label': label,
            'color': color,
            'point_size': point_size,
            'alpha': alpha,
        })

    def add_arrow(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: str = 'blue',
        label: Optional[str] = None,
        linewidth: float = 2.0,
    ) -> None:
        """Add an arrow on the Bloch sphere."""
        self._arrows.append({
            'start': start,
            'end': end,
            'color': color,
            'label': label,
            'linewidth': linewidth,
        })

    def add_trail(self, states: List[np.ndarray], color: str = 'green', alpha: float = 0.5) -> None:
        """Add a trail (sequence of states) on the Bloch sphere."""
        trail = [self.state_to_bloch(s) for s in states]
        self._trails.append({'points': trail, 'color': color, 'alpha': alpha})

    def show(
        self,
        filename: Optional[str] = None,
        dpi: int = 150,
        view_angles: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Render the Bloch sphere.

        Parameters
        ----------
        filename : Optional[str]
            Save to file.
        dpi : int
            DPI for saved figure.
        view_angles : Optional[Tuple[float, float]]
            Elevation and azimuth for 3D view.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Draw sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)

        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray',
                         alpha=0.2, linewidth=0.5)

        # Draw axes
        ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', arrow_length_ratio=0.1, linewidth=1.5)

        ax.text(1.4, 0, 0, 'X', fontsize=12, color='red')
        ax.text(0, 1.4, 0, 'Y', fontsize=12, color='green')
        ax.text(0, 0, 1.4, 'Z', fontsize=12, color='blue')

        # Draw great circles
        theta_gc = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta_gc), np.sin(theta_gc), np.zeros_like(theta_gc),
               'gray', alpha=0.3, linewidth=0.5)
        ax.plot(np.cos(theta_gc), np.zeros_like(theta_gc), np.sin(theta_gc),
               'gray', alpha=0.3, linewidth=0.5)
        ax.plot(np.zeros_like(theta_gc), np.cos(theta_gc), np.sin(theta_gc),
               'gray', alpha=0.3, linewidth=0.5)

        # Plot states
        for s in self._states:
            bx, by, bz = s['bloch']
            ax.scatter([bx], [by], [bz], c=s['color'], s=s['point_size'],
                      alpha=s['alpha'], zorder=5)
            if s['label']:
                ax.text(bx * 1.15, by * 1.15, bz * 1.15, s['label'],
                       fontsize=9, color=s['color'])

        # Plot trails
        for trail in self._trails:
            pts = np.array(trail['points'])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=trail['color'], alpha=trail['alpha'], linewidth=1.5)

        # Plot arrows
        for arrow in self._arrows:
            s = arrow['start']
            e = arrow['end']
            ax.quiver(s[0], s[1], s[2],
                     e[0] - s[0], e[1] - s[1], e[2] - s[2],
                     color=arrow['color'], linewidth=arrow['linewidth'])

        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_box_aspect([1, 1, 1])

        if view_angles:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])

        ax.set_title(self.title or 'Bloch Sphere', fontsize=14, pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False)

        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')

        return fig

    def clear(self) -> None:
        """Clear all states, arrows, and trails."""
        self._states.clear()
        self._arrows.clear()
        self._trails.clear()
