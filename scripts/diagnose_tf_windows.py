"""TensorFlow DLL diagnostic for Windows.

Run from cmd:
    python scripts/diagnose_tf_windows.py

Checks, in order of how often they explain ``0x45A - DLL initialization
routine failed`` on Windows:
  1. 64-bit vs 32-bit Python (TensorFlow ships 64-bit wheels only)
  2. Microsoft Visual C++ Redistributable DLLs (msvcp140 / vcruntime140 /
     vcruntime140_1) - present AND loadable
  3. CPU model (old CPUs without AVX cannot run official TF wheels)
  4. numpy / TensorFlow / QuantumFlow imports with full tracebacks
"""
import os
import platform
import subprocess
import sys
import traceback

BULLET = "  "


def main() -> int:
    print("=== QuantumFlow TensorFlow DLL diagnostic ===\n")

    # -- 1. Python interpreter ------------------------------------------------
    print("[1] Python")
    print(f"{BULLET}version : {sys.version.split()[0]}")
    bits = platform.architecture()[0]
    print(f"{BULLET}bitness : {bits}")
    if bits == "32bit":
        print(f"{BULLET}!! 32-bit Python CANNOT run TensorFlow. Install 64-bit")
        print("   Python (or the 64-bit Anaconda) and reinstall the stack.")
        return 1
    print(f"{BULLET}exe     : {sys.executable}")

    # -- 2. CPU ---------------------------------------------------------------
    print("\n[2] CPU")
    cpu_name = ""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            cpu_name = " ".join(out.split()).replace("Name ", "", 1).strip()
        except Exception as exc:  # pragma: no cover - wmic missing on some SKUs
            print(f"{BULLET}(wmic unavailable: {exc})")
    print(f"{BULLET}model   : {cpu_name or 'unknown'}")
    print(f"{BULLET}note    : official TensorFlow wheels require a CPU with")
    print("            AVX (any Intel Core from ~2011 / AMD Bulldozer onward).")

    # -- 3. Visual C++ Redistributable ----------------------------------------
    print("\n[3] Microsoft Visual C++ Redistributable (2015-2022 x64)")
    problems = []
    if os.name != "nt":
        print(f"{BULLET}(non-Windows system: this section is Windows-only)")
    else:
        import ctypes

        sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                             "System32")
        for dll in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            path = os.path.join(sys32, dll)
            if not os.path.exists(path):
                print(f"{BULLET}{dll} : MISSING  <-- install/repair the VC++ "
                      "redistributable (see below)")
                problems.append(dll)
                continue
            try:
                ctypes.WinDLL(path)
                print(f"{BULLET}{dll} : found and loads OK")
            except OSError as exc:
                print(f"{BULLET}{dll} : found but FAILED to load: {exc}")
                problems.append(dll)
        if problems:
            print(f"{BULLET}fix     : download & run "
                  "https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("            (choose 'Repair' if already installed), then "
                  "REBOOT and re-run this script.")

    # -- 4. numpy -------------------------------------------------------------
    print("\n[4] numpy")
    try:
        import numpy
        print(f"{BULLET}numpy {numpy.__version__} imports OK")
    except Exception:
        traceback.print_exc()

    # -- 5. TensorFlow --------------------------------------------------------
    print("\n[5] TensorFlow")
    try:
        import tensorflow as tf
        print(f"{BULLET}TensorFlow {tf.__version__} imports OK")
    except Exception:
        print(f"{BULLET}TensorFlow FAILED to import:")
        traceback.print_exc()
        print(f"{BULLET}if the traceback shows the _pywrap DLL error, work")
        print("            through: (a) VC++ redist above, (b) reinstall:")
        print("            pip uninstall tensorflow keras -y && pip cache purge")
        print("            && pip install tensorflow, (c) try")
        print("            set KMP_DUPLICATE_LIB_OK=1 before importing.")

    # -- 6. QuantumFlow -------------------------------------------------------
    print("\n[6] QuantumFlow")
    try:
        import quantumflow as qf
        print(f"{BULLET}QuantumFlow {qf.__version__} imports OK "
              "(core works even without TensorFlow)")
    except Exception:
        traceback.print_exc()

    print("\n=== done — paste this entire output when asking for help ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
