"""
PRIMO Backend — numpy ↔ torch device abstraction.

Provides a unified interface for linear algebra operations that can run
on either CPU (numpy) or GPU (torch). The rest of the codebase imports
from here instead of calling numpy.linalg or torch.linalg directly.

Usage:
    from primo.backend import get_backend
    B = get_backend()          # reads DEVICE from config.py
    B = get_backend("cpu")     # force CPU
    vals, vecs = B.eigh(matrix)
    result = B.to_numpy(vals)  # always returns a numpy array
"""

import numpy as np
from primo.config import DEVICE


def get_backend(device=None):
    """Return a Backend instance for the given device string.

    Parameters
    ----------
    device : str or None
        "cuda", "cpu", or None (uses config.DEVICE).
        Falls back to CPU if torch/CUDA unavailable.
    """
    device = device or DEVICE
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return TorchBackend(torch.device("cuda"))
        except ImportError:
            pass
        # Fallback: requested CUDA but unavailable
        import warnings
        warnings.warn("CUDA requested but unavailable, falling back to CPU")
    return NumpyBackend()


class NumpyBackend:
    """Pure-numpy backend (CPU)."""

    name = "numpy"

    def is_cuda(self):
        return False

    def device_name(self):
        return "cpu (numpy)"

    # --- Device transfer ---

    def to_device(self, array):
        """Ensure input is a numpy ndarray."""
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array, dtype=np.float64)

    def to_numpy(self, array):
        """Return a numpy ndarray (identity for this backend)."""
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array)

    # --- Construction ---

    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def array(self, data, dtype=np.float64):
        return np.asarray(data, dtype=dtype)

    def arange(self, *args, dtype=np.float64):
        return np.arange(*args, dtype=dtype)

    # --- Decompositions ---

    def eigh(self, matrix):
        """Symmetric eigendecomposition. Returns (eigenvalues, eigenvectors)."""
        return np.linalg.eigh(matrix)

    def eigvalsh(self, matrix):
        """Symmetric eigenvalues only."""
        return np.linalg.eigvalsh(matrix)

    def svd(self, matrix, full_matrices=False, compute_uv=True):
        """Singular value decomposition.

        Returns (U, S, Vt) if compute_uv=True, else S only.
        """
        if compute_uv:
            return np.linalg.svd(matrix, full_matrices=full_matrices)
        return np.linalg.svd(matrix, full_matrices=full_matrices, compute_uv=False)

    # --- Matrix operations ---

    def matmul(self, a, b):
        return a @ b

    # --- Element-wise & reductions ---

    def mean(self, array, axis=None):
        return np.mean(array, axis=axis)

    def std(self, array, axis=None):
        return np.std(array, axis=axis)

    def sum(self, array, axis=None):
        return np.sum(array, axis=axis)

    def sqrt(self, array):
        return np.sqrt(array)

    def log(self, array):
        return np.log(array)

    def abs(self, array):
        return np.abs(array)

    # --- Fitting (CPU-only, operates on numpy) ---

    def polyfit(self, x, y, degree):
        x_np = self.to_numpy(x)
        y_np = self.to_numpy(y)
        return np.polyfit(x_np, y_np, degree)

    def polyval(self, coeffs, x):
        x_np = self.to_numpy(x)
        coeffs_np = self.to_numpy(coeffs) if not isinstance(coeffs, np.ndarray) else coeffs
        return np.polyval(coeffs_np, x_np)

    # --- Utility ---

    def norm(self, array, axis=None):
        return np.linalg.norm(array, axis=axis)


class TorchBackend:
    """PyTorch GPU backend."""

    name = "torch"

    def __init__(self, device):
        import torch
        self._torch = torch
        self._device = device

    def is_cuda(self):
        return True

    def device_name(self):
        import torch
        return torch.cuda.get_device_name(self._device)

    # --- Device transfer ---

    def to_device(self, array):
        """Convert numpy array (or list) to a torch tensor on the GPU."""
        torch = self._torch
        if isinstance(array, torch.Tensor):
            return array.to(self._device)
        arr = np.asarray(array, dtype=np.float64)
        return torch.from_numpy(arr).to(self._device)

    def to_numpy(self, tensor):
        """Convert a torch tensor to a numpy array."""
        torch = self._torch
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    # --- Construction ---

    def zeros(self, shape, dtype=np.float64):
        torch = self._torch
        torch_dtype = self._numpy_to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=torch_dtype, device=self._device)

    def array(self, data, dtype=np.float64):
        torch = self._torch
        arr = np.asarray(data, dtype=dtype)
        return torch.from_numpy(arr).to(self._device)

    def arange(self, *args, dtype=np.float64):
        torch = self._torch
        torch_dtype = self._numpy_to_torch_dtype(dtype)
        return torch.arange(*args, dtype=torch_dtype, device=self._device)

    # --- Decompositions ---

    def eigh(self, matrix):
        """Symmetric eigendecomposition on GPU."""
        torch = self._torch
        t = self.to_device(matrix)
        vals, vecs = torch.linalg.eigh(t)
        return vals, vecs

    def eigvalsh(self, matrix):
        """Symmetric eigenvalues only on GPU."""
        torch = self._torch
        t = self.to_device(matrix)
        return torch.linalg.eigvalsh(t)

    def svd(self, matrix, full_matrices=False, compute_uv=True):
        """SVD on GPU."""
        torch = self._torch
        t = self.to_device(matrix)
        if compute_uv:
            U, S, Vt = torch.linalg.svd(t, full_matrices=full_matrices)
            return U, S, Vt
        return torch.linalg.svdvals(t)

    # --- Matrix operations ---

    def matmul(self, a, b):
        torch = self._torch
        a_t = self.to_device(a) if not isinstance(a, torch.Tensor) else a
        b_t = self.to_device(b) if not isinstance(b, torch.Tensor) else b
        return a_t @ b_t

    # --- Element-wise & reductions ---

    def mean(self, array, axis=None):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        if axis is None:
            return t.mean()
        return t.mean(dim=axis)

    def std(self, array, axis=None):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        # Use correction=0 to match numpy's default (population std)
        if axis is None:
            return t.std(correction=0)
        return t.std(dim=axis, correction=0)

    def sum(self, array, axis=None):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        if axis is None:
            return t.sum()
        return t.sum(dim=axis)

    def sqrt(self, array):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        return torch.sqrt(t)

    def log(self, array):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        return torch.log(t)

    def abs(self, array):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        return torch.abs(t)

    # --- Fitting (falls back to numpy — polyfit has no GPU benefit) ---

    def polyfit(self, x, y, degree):
        x_np = self.to_numpy(x)
        y_np = self.to_numpy(y)
        return np.polyfit(x_np, y_np, degree)

    def polyval(self, coeffs, x):
        x_np = self.to_numpy(x)
        coeffs_np = self.to_numpy(coeffs) if not isinstance(coeffs, np.ndarray) else coeffs
        return np.polyval(coeffs_np, x_np)

    # --- Utility ---

    def norm(self, array, axis=None):
        torch = self._torch
        t = self.to_device(array) if not isinstance(array, torch.Tensor) else array
        if axis is None:
            return torch.linalg.norm(t)
        return torch.linalg.norm(t, dim=axis)

    def _numpy_to_torch_dtype(self, dtype):
        """Map numpy dtype to torch dtype."""
        torch = self._torch
        mapping = {
            np.float64: torch.float64,
            np.float32: torch.float32,
            np.int64: torch.int64,
            np.int32: torch.int32,
        }
        return mapping.get(dtype, torch.float64)
