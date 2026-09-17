#!/usr/bin/env python
"""Setup script for QuantumFlow.

Builds the optional Cython acceleration kernels when Cython and a C
compiler are available. The package is fully functional from pure Python —
if any extension cannot be built (missing Cython, no compiler, unusual
platform) it is skipped with a warning instead of failing the install.
"""

import os
import warnings

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# numpy is a *runtime* dependency declared in pyproject.toml; the
# isolated environment pip uses to build wheels does not necessarily
# contain it (it only installs [build-system] requires).  Import it
# defensively so metadata generation never crashes — the Cython
# kernels are optional and simply get skipped without numpy headers.
try:
    import numpy as np
except ImportError:  # pragma: no cover - minimal build environments
    np = None

HERE = os.path.abspath(os.path.dirname(__file__))

# Cython extensions for performance-critical operations. All optional.
EXTENSION_SPECS = [
    (
        "quantumflow.core._fast_gates",
        ["quantumflow/core/_fast_gates.pyx"],
    ),
    (
        "quantumflow.simulation._fast_simulator",
        ["quantumflow/simulation/_fast_simulator.pyx"],
    ),
    (
        "quantumflow.utils._fast_math",
        ["quantumflow/utils/_fast_math.pyx"],
    ),
]


class _OptionalBuildExt(_build_ext):
    """``build_ext`` that never fails the install.

    If an individual extension cannot be compiled (missing headers, no
    compiler, unsupported platform) it is dropped from the build with a
    warning. QuantumFlow is fully functional without the extensions.
    """

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - build environment
            warnings.warn(
                f"Optional Cython extensions could not be built ({exc}); "
                "QuantumFlow will use the pure-Python fallbacks.",
                stacklevel=2,
            )

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - build environment
            warnings.warn(
                f"Skipping optional extension {ext.name}: {exc}",
                stacklevel=2,
            )
            # Remove the extension so the remaining build steps (and
            # install) succeed without it.
            if ext in self.extensions:
                self.extensions.remove(ext)


def _sources_exist(sources):
    return all(os.path.exists(os.path.join(HERE, s)) for s in sources)


def get_extensions():
    try:
        from Cython.Build import cythonize
    except ImportError:
        warnings.warn(
            "Cython is not installed: skipping the optional acceleration "
            "kernels (pip install Cython to build them).",
            stacklevel=2,
        )
        return []

    if np is None:
        warnings.warn(
            "numpy is not available in the build environment: skipping the "
            "optional acceleration kernels (pure-Python fallbacks will be "
            "used at runtime).",
            stacklevel=2,
        )
        return []

    extensions = []
    for name, sources in EXTENSION_SPECS:
        if not _sources_exist(sources):
            warnings.warn(
                f"Sources for optional extension {name} not found; skipping.",
                stacklevel=2,
            )
            continue
        extensions.append(
            Extension(
                name,
                sources,
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        )

    if not extensions:
        return []

    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )


setup(
    ext_modules=get_extensions(),
    packages=find_packages(include=["quantumflow", "quantumflow.*"]),
    cmdclass={"build_ext": _OptionalBuildExt},
    zip_safe=False,
)
