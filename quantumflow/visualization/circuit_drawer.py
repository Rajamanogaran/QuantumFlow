"""
Quantum Circuit Drawing
========================

Provides visualization of quantum circuits in text, matplotlib,
and LaTeX formats.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class OutputFormat(Enum):
    """Output format for circuit drawing."""
    TEXT = "text"
    MATPLOTLIB = "matplotlib"
    LATEX = "latex"
    SVG = "svg"


class CircuitDrawer:
    """
    Quantum circuit visualization tool.

    Renders quantum circuits in multiple formats: ASCII art,
    matplotlib, and LaTeX.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to draw.
    scale : float
        Scale factor for the output.
    wire_order : Optional[List[int]]
        Custom wire ordering. If None, uses natural ordering.

    Examples
    --------
    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.rz(np.pi/4, 2)
    >>> drawer = CircuitDrawer(qc)
    >>> print(drawer.draw_text())
    """

    # Gate display names and Unicode symbols
    GATE_SYMBOLS = {
        'h': 'H', 'x': 'X', 'y': 'Y', 'z': 'Z',
        's': 'S', 'sdg': 'S†', 't': 'T', 'tdg': 'T†',
        'sx': '√X', 'sxdg': '√X†',
        'rx': 'Rx', 'ry': 'Ry', 'rz': 'Rz',
        'cx': '●', 'cz': '●', 'swap': '×',
        'ccx': '●', 'measure': 'M', 'barrier': '│',
        'reset': '|0⟩',
        'u1': 'U1', 'u2': 'U2', 'u3': 'U3',
        'u': 'U', 'p': 'P', 'rxx': 'RXX', 'ryy': 'RYY',
        'rzz': 'RZZ', 'rzx': 'RZX',
    }

    # Box width for multi-character gates
    BOX_GATES = {'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'u', 'p', 'rxx', 'ryy', 'rzz', 'rzx'}

    def __init__(
        self,
        circuit: Any,
        scale: float = 1.0,
        wire_order: Optional[List[int]] = None,
    ) -> None:
        self.circuit = circuit
        self.scale = scale
        self.wire_order = wire_order or list(range(circuit.n_qubits if hasattr(circuit, 'n_qubits') else circuit.width()))

        # Extract operations
        self._operations = []
        if hasattr(circuit, '_operations'):
            self._operations = circuit._operations
        elif hasattr(circuit, 'data'):
            self._operations = circuit.data

    def draw_text(
        self,
        label: Optional[str] = None,
        line_width: int = 80,
        encoding: str = 'unicode',
    ) -> str:
        """
        Draw circuit as ASCII/Unicode art.

        Parameters
        ----------
        label : Optional[str]
            Circuit label.
        line_width : int
            Maximum line width.
        encoding : str
            'unicode' or 'ascii'.

        Returns
        -------
        str
            ASCII art representation of the circuit.
        """
        n_qubits = len(self.wire_order)
        if n_qubits == 0:
            return "(empty circuit)"

        # Initialize wire lines
        lines = ['─' * line_width for _ in range(n_qubits)]
        wire_labels = [f'q{i}: ' for i in self.wire_order]

        col = 4  # Starting column (after label)

        for op in self._operations:
            gate_name = self._get_gate_name(op)
            qubits = self._get_qubits(op)
            params = self._get_params(op)

            if gate_name == 'barrier':
                for q in qubits:
                    wire_idx = self._wire_index(q)
                    lines[wire_idx] = lines[wire_idx][:col] + '┃' + lines[wire_idx][col+1:]
                col += 1
                continue

            display_name = self._format_gate(gate_name, params, encoding)
            gate_width = max(len(display_name), 1)

            if len(qubits) == 1:
                wire_idx = self._wire_index(qubits[0])
                if gate_name == 'measure':
                    lines[wire_idx] = (
                        lines[wire_idx][:col] + '┤M├' + lines[wire_idx][col+3:]
                    )
                    col = max(col + 3, col + gate_width)
                else:
                    lines[wire_idx] = (
                        lines[wire_idx][:col] + f'┤{display_name}├' +
                        lines[wire_idx][col + gate_width + 2:]
                    )
                    col += gate_width + 2

            elif len(qubits) == 2:
                wire_idx_0 = self._wire_index(qubits[0])
                wire_idx_1 = self._wire_index(qubits[1])

                if gate_name in ('cx', 'ccx', 'cz', 'swap'):
                    # Control-target representation
                    if gate_name == 'swap':
                        symbol_0 = '×'
                        symbol_1 = '×'
                    elif gate_name == 'cz':
                        symbol_0 = '●'
                        symbol_1 = '●'
                    else:
                        symbol_0 = '●'
                        symbol_1 = display_name if display_name != 'X' else '⊕'

                    lines[wire_idx_0] = (
                        lines[wire_idx_0][:col] + symbol_0 +
                        lines[wire_idx_0][col+1:]
                    )
                    # Vertical line
                    for w in range(min(wire_idx_0, wire_idx_1) + 1,
                                   max(wire_idx_0, wire_idx_1)):
                        if lines[w][col] == '─':
                            lines[w] = lines[w][:col] + '│' + lines[w][col+1:]
                    lines[wire_idx_1] = (
                        lines[wire_idx_1][:col] + symbol_1 +
                        lines[wire_idx_1][col+1:]
                    )
                    col = max(col + 1, col + gate_width)

            elif len(qubits) >= 3:
                for i, q in enumerate(qubits):
                    wire_idx = self._wire_index(q)
                    if i == 0:
                        lines[wire_idx] = (
                            lines[wire_idx][:col] + '●' + lines[wire_idx][col+1:]
                        )
                    elif i == len(qubits) - 1:
                        target_sym = display_name if display_name else '⊕'
                        lines[wire_idx] = (
                            lines[wire_idx][:col] + target_sym +
                            lines[wire_idx][col+1:]
                        )
                    else:
                        lines[wire_idx] = (
                            lines[wire_idx][:col] + '●' + lines[wire_idx][col+1:]
                        )

                min_w = min(self._wire_index(q) for q in qubits)
                max_w = max(self._wire_index(q) for q in qubits)
                for w in range(min_w + 1, max_w):
                    lines[w] = lines[w][:col] + '│' + lines[w][col+1:]

                col += 2

        # Add wire labels
        result_lines = []
        for i, (line, label) in enumerate(zip(lines, wire_labels)):
            result_lines.append(f'{label}{line}')

        if label:
            header = f'  {label}'
            result_lines.insert(0, header)
            result_lines.insert(1, '  ' + '═' * (line_width // 2))

        return '\n'.join(result_lines)

    def draw_matplotlib(
        self,
        figsize: Tuple[float, float] = (12, 6),
        title: Optional[str] = None,
        style: str = 'default',
        filename: Optional[str] = None,
        dpi: int = 150,
    ) -> Any:
        """
        Draw circuit using matplotlib.

        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size in inches.
        title : Optional[str]
            Plot title.
        style : str
            Plot style.
        filename : Optional[str]
            Save to file. If None, displays.
        dpi : int
            DPI for saved figures.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        n_qubits = len(self.wire_order)
        n_ops = len(self._operations)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(-1, max(n_ops * 1.5, 10))
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.invert_yaxis()

        # Draw wires
        for i in range(n_qubits):
            ax.axhline(y=i, color='black', linewidth=0.5, xmin=0.05, xmax=0.95)
            ax.text(-0.5, i, f'q{self.wire_order[i]}', ha='right', va='center', fontsize=10)

        # Draw gates
        x_pos = 1.0
        for op in self._operations:
            gate_name = self._get_gate_name(op)
            qubits = self._get_qubits(op)

            if gate_name == 'barrier':
                for q in qubits:
                    w = self._wire_index(q)
                    ax.axvline(x=x_pos, ymin=(w) / n_qubits,
                             ymax=(w + 1) / n_qubits,
                             color='gray', linewidth=1, linestyle='--')
                x_pos += 0.5
                continue

            if len(qubits) == 1:
                w = self._wire_index(qubits[0])
                color = self._gate_color(gate_name)
                rect = patches.FancyBboxPatch(
                    (x_pos - 0.3, w - 0.25), 0.6, 0.5,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor='black', linewidth=1.5,
                    alpha=0.8,
                )
                ax.add_patch(rect)
                ax.text(x_pos, w, gate_name.upper(), ha='center', va='center', fontsize=8)
                x_pos += 1.5

            elif len(qubits) == 2:
                w0 = self._wire_index(qubits[0])
                w1 = self._wire_index(qubits[1])

                # Control dot
                ax.plot(x_pos, w0, 'ko', markersize=8)

                # Vertical line
                ax.plot([x_pos, x_pos], [w0, w1], 'k-', linewidth=1.5)

                # Target
                if gate_name == 'swap':
                    ax.plot(x_pos, w1, 'x', markersize=8, markeredgewidth=2)
                elif gate_name == 'cz':
                    ax.plot(x_pos, w1, 'ko', markersize=8)
                else:
                    circle = patches.Circle((x_pos, w1), 0.25, fill=False,
                                           edgecolor='black', linewidth=1.5)
                    ax.add_patch(circle)
                    ax.plot([x_pos - 0.2, x_pos + 0.2], [w1, w1], 'k-', linewidth=1.5)
                    ax.plot([x_pos, x_pos], [w1 - 0.2, w1 + 0.2], 'k-', linewidth=1.5)

                x_pos += 1.5

        ax.set_title(title or 'Quantum Circuit')
        ax.axis('off')
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')

        return fig

    def draw_latex(self, label: Optional[str] = None) -> str:
        """
        Generate LaTeX Qcircuit code for the circuit.

        Parameters
        ----------
        label : Optional[str]
            Circuit label.

        Returns
        -------
        str
            LaTeX code string.
        """
        n_qubits = len(self.wire_order)

        lines = []
        if label:
            lines.append(f'\\text{{{label}}}:')

        lines.append('\\Qcircuit @C=1em @R=.7em {')
        for i in range(n_qubits):
            lines.append(f'  \\lstick{{q{self.wire_order[i]}}} & \\qw &')

        # Placeholder for gate rendering
        for op in self._operations:
            gate_name = self._get_gate_name(op)
            qubits = self._get_qubits(op)
            # Simplified LaTeX output
            if len(qubits) == 1:
                lines.append(f'  & \\gate{{{gate_name.upper()}}} &')
            elif len(qubits) == 2 and gate_name in ('cx', 'cz'):
                ctrl = self._wire_index(qubits[0])
                tgt = self._wire_index(qubits[1])
                for i in range(n_qubits):
                    if i == ctrl:
                        lines.append('  & \\ctrl{1} &')
                    elif i == tgt:
                        if gate_name == 'cx':
                            lines.append('  & \\targ &')
                        else:
                            lines.append('  & \\control\\qw &')

        lines.append('}')
        return '\n'.join(lines)

    def _wire_index(self, qubit: Any) -> int:
        """Get wire index for a qubit."""
        if isinstance(qubit, int):
            if qubit in self.wire_order:
                return self.wire_order.index(qubit)
            return qubit
        q_idx = getattr(qubit, 'index', getattr(qubit, '_index', None))
        if q_idx is not None and q_idx in self.wire_order:
            return self.wire_order.index(q_idx)
        return 0

    def _get_gate_name(self, op: Any) -> str:
        """Extract gate name from operation."""
        if hasattr(op, 'name'):
            return op.name.lower()
        if hasattr(op, 'gate'):
            name = getattr(op.gate, 'name', str(type(op.gate).__name__))
            return name.lower()
        return str(type(op).__name__).lower().replace('gate', '')

    def _get_qubits(self, op: Any) -> List[int]:
        """Extract qubit indices from operation."""
        if hasattr(op, 'qubits'):
            return list(op.qubits)
        if hasattr(op, 'indices'):
            return list(op.indices)
        return []

    def _get_params(self, op: Any) -> List[float]:
        """Extract parameters from operation."""
        if hasattr(op, 'params'):
            return list(op.params)
        if hasattr(op, 'parameters'):
            params = list(op.parameters.values())
            return [float(p) for p in params]
        return []

    def _format_gate(self, name: str, params: List[float], encoding: str = 'unicode') -> str:
        """Format gate name with parameters."""
        symbol = self.GATE_SYMBOLS.get(name, name.upper())

        if params and name in self.BOX_GATES:
            if len(params) == 1:
                return f'{symbol}({params[0]:.2f})'
            elif len(params) == 3:
                return f'{symbol}({params[0]:.2f},{params[1]:.2f},{params[2]:.2f})'

        return symbol

    @staticmethod
    def _gate_color(gate_name: str) -> str:
        """Get color for gate visualization."""
        colors = {
            'h': '#4FC3F7', 'x': '#EF5350', 'y': '#AB47BC', 'z': '#66BB6A',
            'rx': '#4FC3F7', 'ry': '#AB47BC', 'rz': '#66BB6A',
            'measure': '#FFD54F', 'cx': '#FF7043', 'swap': '#8D6E63',
        }
        return colors.get(gate_name, '#90A4AE')
