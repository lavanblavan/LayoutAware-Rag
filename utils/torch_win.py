"""Windows helper so PyTorch can load c10.dll after OpenCV / other native libs."""
from __future__ import annotations

import os
import sys


def bootstrap_torch() -> None:
    if sys.platform != "win32":
        return
    if getattr(bootstrap_torch, "_done", False):
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return
        torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
        # Reduce OpenMP / MKL clashes when cv2 and torch share a process.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    except Exception:
        pass
    bootstrap_torch._done = True
