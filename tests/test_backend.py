"""
Tests for primo.backend — numpy ↔ torch device abstraction.

Verifies:
1. NumpyBackend works correctly for all operations
2. TorchBackend (if available) produces identical results within tolerance
3. CUDA detection and fallback behavior
4. Round-trip: numpy → device → numpy preserves values
"""

import numpy as np
import pytest

from primo.backend import get_backend, NumpyBackend, TorchBackend


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def cpu():
    """Always-available numpy backend."""
    return get_backend("cpu")


@pytest.fixture
def gpu():
    """Torch/CUDA backend, skipped if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not installed")
    return get_backend("cuda")


def both_backends(request):
    """Parametrize over available backends."""
    backends = [get_backend("cpu")]
    try:
        import torch
        if torch.cuda.is_available():
            backends.append(get_backend("cuda"))
    except ImportError:
        pass
    return backends


@pytest.fixture(params=["cpu", "cuda"], ids=["cpu", "cuda"])
def backend(request):
    """Parametrized fixture: test on each available backend."""
    if request.param == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")
    return get_backend(request.param)


# ── Helpers ───────────────────────────────────────────────────────────

ATOL = 1e-10  # absolute tolerance for float64 comparisons

def to_np(B, x):
    """Convert backend result to numpy for comparison."""
    return B.to_numpy(x)


# ── Test: Backend selection and properties ────────────────────────────

class TestBackendSelection:
    def test_cpu_returns_numpy(self):
        B = get_backend("cpu")
        assert isinstance(B, NumpyBackend)
        assert not B.is_cuda()
        assert "cpu" in B.device_name()

    def test_cuda_returns_torch_or_warns(self):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if has_cuda:
            B = get_backend("cuda")
            assert isinstance(B, TorchBackend)
            assert B.is_cuda()
            assert len(B.device_name()) > 0  # should report GPU name
        else:
            with pytest.warns(UserWarning, match="CUDA requested but unavailable"):
                B = get_backend("cuda")
            assert isinstance(B, NumpyBackend)

    def test_none_uses_config(self):
        # get_backend(None) should return something valid regardless
        B = get_backend(None)
        assert hasattr(B, "eigh")


# ── Test: Device transfer round-trip ──────────────────────────────────

class TestDeviceTransfer:
    def test_to_device_and_back(self, backend):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        on_device = backend.to_device(arr)
        back = backend.to_numpy(on_device)
        np.testing.assert_array_equal(back, arr)

    def test_to_device_from_list(self, backend):
        data = [[1.0, 2.0], [3.0, 4.0]]
        on_device = backend.to_device(data)
        back = backend.to_numpy(on_device)
        np.testing.assert_array_equal(back, np.array(data))

    def test_to_numpy_idempotent(self, backend):
        arr = np.array([1.0, 2.0, 3.0])
        result = backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)


# ── Test: Construction ────────────────────────────────────────────────

class TestConstruction:
    def test_zeros(self, backend):
        z = backend.zeros((3, 4))
        result = to_np(backend, z)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.zeros((3, 4)))

    def test_array(self, backend):
        data = [[1.0, 2.0], [3.0, 4.0]]
        a = backend.array(data)
        result = to_np(backend, a)
        np.testing.assert_array_equal(result, np.array(data))

    def test_arange(self, backend):
        a = backend.arange(5, dtype=np.float64)
        result = to_np(backend, a)
        np.testing.assert_array_equal(result, np.arange(5, dtype=np.float64))


# ── Test: Eigendecomposition ──────────────────────────────────────────

class TestEigh:
    def test_symmetric_eigenvalues(self, backend):
        """eigh on a known symmetric matrix."""
        # Laplacian of a triangle graph
        L = np.array([
            [ 2., -1., -1.],
            [-1.,  2., -1.],
            [-1., -1.,  2.],
        ])
        vals, vecs = backend.eigh(L)
        vals_np = to_np(backend, vals)
        vecs_np = to_np(backend, vecs)

        # Known eigenvalues of K3 Laplacian: 0, 3, 3
        expected_vals = np.array([0.0, 3.0, 3.0])
        np.testing.assert_allclose(vals_np, expected_vals, atol=ATOL)

        # Eigenvectors should be orthonormal
        eye = vecs_np.T @ vecs_np
        np.testing.assert_allclose(eye, np.eye(3), atol=ATOL)

    def test_eigvalsh_matches_eigh(self, backend):
        """eigvalsh returns same eigenvalues as eigh."""
        A = np.array([[4., 2.], [2., 3.]])
        vals_only = backend.eigvalsh(A)
        vals_full, _ = backend.eigh(A)
        np.testing.assert_allclose(
            to_np(backend, vals_only),
            to_np(backend, vals_full),
            atol=ATOL,
        )


# ── Test: SVD ─────────────────────────────────────────────────────────

class TestSVD:
    def test_full_svd(self, backend):
        """SVD reconstructs the original matrix."""
        A = np.array([[1., 2., 3.], [4., 5., 6.]])
        U, S, Vt = backend.svd(A, full_matrices=False)
        U_np = to_np(backend, U)
        S_np = to_np(backend, S)
        Vt_np = to_np(backend, Vt)

        reconstructed = U_np @ np.diag(S_np) @ Vt_np
        np.testing.assert_allclose(reconstructed, A, atol=ATOL)

    def test_svd_values_only(self, backend):
        """compute_uv=False returns singular values only."""
        A = np.array([[1., 0.], [0., 2.], [0., 0.]])
        S = backend.svd(A, compute_uv=False)
        S_np = to_np(backend, S)
        np.testing.assert_allclose(sorted(S_np, reverse=True), [2.0, 1.0], atol=ATOL)

    def test_subspace_cosine_pipeline(self, backend):
        """Replicate the subspace_cosine computation from the oracle."""
        np.random.seed(42)
        X1 = np.random.randn(10, 5)
        X2 = X1 + 0.01 * np.random.randn(10, 5)  # nearly identical

        U1, _, _ = backend.svd(backend.to_device(X1), full_matrices=False)
        U2, _, _ = backend.svd(backend.to_device(X2), full_matrices=False)

        k = 3
        # Slice first k columns
        U1_np = to_np(backend, U1)[:, :k]
        U2_np = to_np(backend, U2)[:, :k]

        gram = backend.matmul(
            backend.to_device(U1_np.T),
            backend.to_device(U2_np),
        )
        S = backend.svd(gram, compute_uv=False)
        cosine = float(to_np(backend, backend.mean(S)))

        # Nearly identical inputs → cosine ≈ 1.0
        assert cosine > 0.99, f"Expected cosine > 0.99, got {cosine}"


# ── Test: Matrix operations ───────────────────────────────────────────

class TestMatmul:
    def test_matmul(self, backend):
        A = np.array([[1., 2.], [3., 4.]])
        B = np.array([[5., 6.], [7., 8.]])
        result = to_np(backend, backend.matmul(
            backend.to_device(A), backend.to_device(B)
        ))
        expected = A @ B
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_matmul_non_square(self, backend):
        A = np.random.randn(3, 5)
        B = np.random.randn(5, 2)
        result = to_np(backend, backend.matmul(
            backend.to_device(A), backend.to_device(B)
        ))
        np.testing.assert_allclose(result, A @ B, atol=ATOL)


# ── Test: Reductions ──────────────────────────────────────────────────

class TestReductions:
    def test_mean(self, backend):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = float(to_np(backend, backend.mean(backend.to_device(arr))))
        assert abs(result - 2.5) < ATOL

    def test_std(self, backend):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = float(to_np(backend, backend.std(backend.to_device(arr))))
        expected = float(np.std(arr))  # population std
        assert abs(result - expected) < ATOL

    def test_sum(self, backend):
        arr = np.array([1.0, 2.0, 3.0])
        result = float(to_np(backend, backend.sum(backend.to_device(arr))))
        assert abs(result - 6.0) < ATOL

    def test_mean_axis(self, backend):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_np(backend, backend.mean(backend.to_device(arr), axis=0))
        np.testing.assert_allclose(result, [2.0, 3.0], atol=ATOL)


# ── Test: Element-wise operations ─────────────────────────────────────

class TestElementwise:
    def test_sqrt(self, backend):
        arr = np.array([1.0, 4.0, 9.0, 16.0])
        result = to_np(backend, backend.sqrt(backend.to_device(arr)))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0], atol=ATOL)

    def test_log(self, backend):
        arr = np.array([1.0, np.e, np.e**2])
        result = to_np(backend, backend.log(backend.to_device(arr)))
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=ATOL)

    def test_abs(self, backend):
        arr = np.array([-3.0, 0.0, 5.0])
        result = to_np(backend, backend.abs(backend.to_device(arr)))
        np.testing.assert_allclose(result, [3.0, 0.0, 5.0], atol=ATOL)


# ── Test: Polynomial fitting ─────────────────────────────────────────

class TestPolyfit:
    def test_linear_fit(self, backend):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0  # perfect line: y = 2x + 1
        coeffs = backend.polyfit(x, y, 1)
        np.testing.assert_allclose(coeffs, [2.0, 1.0], atol=ATOL)

    def test_polyval(self, backend):
        coeffs = np.array([2.0, 1.0])  # y = 2x + 1
        x = np.array([0.0, 1.0, 2.0])
        result = backend.polyval(coeffs, x)
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0], atol=ATOL)


# ── Test: Norm ────────────────────────────────────────────────────────

class TestNorm:
    def test_vector_norm(self, backend):
        arr = np.array([3.0, 4.0])
        result = float(to_np(backend, backend.norm(backend.to_device(arr))))
        assert abs(result - 5.0) < ATOL

    def test_matrix_row_norms(self, backend):
        arr = np.array([[3.0, 4.0], [0.0, 1.0]])
        result = to_np(backend, backend.norm(backend.to_device(arr), axis=1))
        np.testing.assert_allclose(result, [5.0, 1.0], atol=ATOL)


# ── Test: Cross-backend consistency ───────────────────────────────────

class TestCrossBackend:
    """Compare numpy and torch backends produce identical results.
    Skipped if CUDA unavailable."""

    @pytest.fixture
    def backends(self):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not installed")
        return get_backend("cpu"), get_backend("cuda")

    def test_eigh_cross(self, backends):
        cpu, gpu = backends
        np.random.seed(123)
        A = np.random.randn(20, 20)
        A = A + A.T  # symmetric

        vals_c, vecs_c = cpu.eigh(A)
        vals_g, vecs_g = gpu.eigh(A)

        np.testing.assert_allclose(
            cpu.to_numpy(vals_c),
            gpu.to_numpy(vals_g),
            atol=1e-8,
        )
        # Eigenvectors may differ in sign; compare absolute values
        np.testing.assert_allclose(
            np.abs(cpu.to_numpy(vecs_c)),
            np.abs(gpu.to_numpy(vecs_g)),
            atol=1e-8,
        )

    def test_svd_cross(self, backends):
        cpu, gpu = backends
        np.random.seed(456)
        A = np.random.randn(15, 5)

        U_c, S_c, Vt_c = cpu.svd(A)
        U_g, S_g, Vt_g = gpu.svd(A)

        # Singular values should match exactly
        np.testing.assert_allclose(
            cpu.to_numpy(S_c),
            gpu.to_numpy(S_g),
            atol=1e-8,
        )

    def test_matmul_cross(self, backends):
        cpu, gpu = backends
        np.random.seed(789)
        A = np.random.randn(10, 8)
        B = np.random.randn(8, 6)

        result_c = cpu.to_numpy(cpu.matmul(A, B))
        result_g = gpu.to_numpy(gpu.matmul(A, B))
        np.testing.assert_allclose(result_c, result_g, atol=1e-8)

    def test_reductions_cross(self, backends):
        cpu, gpu = backends
        arr = np.random.randn(100)

        for op in ["mean", "std", "sum"]:
            val_c = float(cpu.to_numpy(getattr(cpu, op)(arr)))
            val_g = float(gpu.to_numpy(getattr(gpu, op)(arr)))
            assert abs(val_c - val_g) < 1e-8, f"{op} mismatch: {val_c} vs {val_g}"

    def test_full_subspace_cosine_cross(self, backends):
        """The critical I-predicate pipeline must match across backends."""
        cpu, gpu = backends
        np.random.seed(42)
        X1 = np.random.randn(50, 5)
        X2 = X1 + 0.1 * np.random.randn(50, 5)

        def subspace_cosine(B, X1, X2, k=3):
            U1, _, _ = B.svd(B.to_device(X1), full_matrices=False)
            U2, _, _ = B.svd(B.to_device(X2), full_matrices=False)
            U1_k = B.to_numpy(U1)[:, :k]
            U2_k = B.to_numpy(U2)[:, :k]
            gram = B.matmul(B.to_device(U1_k.T), B.to_device(U2_k))
            S = B.svd(gram, compute_uv=False)
            return float(B.to_numpy(B.mean(S)))

        cos_c = subspace_cosine(cpu, X1, X2)
        cos_g = subspace_cosine(gpu, X1, X2)
        assert abs(cos_c - cos_g) < 1e-8, f"Subspace cosine: {cos_c} vs {cos_g}"


# ── Test: CUDA smoke test ─────────────────────────────────────────────

class TestCUDASmoke:
    """Quick GPU functionality check."""

    def test_cuda_available(self):
        """Report CUDA status (informational — does not fail)."""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                mem_gb = mem / (1024**3)
                print(f"\n  CUDA available: {name} ({mem_gb:.1f} GB)")
            else:
                print("\n  CUDA not available (torch installed but no GPU)")
        except ImportError:
            print("\n  PyTorch not installed — CPU-only mode")

    def test_gpu_allocation(self, gpu):
        """Allocate and compute on GPU to verify VRAM works."""
        # Allocate a moderately sized matrix (like a Laplacian for N_MAX=500)
        n = 500
        A = np.random.randn(n, n).astype(np.float64)
        A = A + A.T  # symmetric

        on_gpu = gpu.to_device(A)
        vals = gpu.eigvalsh(on_gpu)
        result = gpu.to_numpy(vals)

        assert result.shape == (n,)
        # Eigenvalues of symmetric matrix should be real
        assert np.all(np.isfinite(result))

    def test_gpu_batch_svd(self, gpu):
        """Simulate batched SVD — the core I-predicate workload."""
        np.random.seed(42)
        batch_size = 8  # GPU_BATCH_SIZE from config
        n, d = 100, 5

        matrices = [np.random.randn(n, d) for _ in range(batch_size)]

        results = []
        for M in matrices:
            U, S, Vt = gpu.svd(M, full_matrices=False)
            results.append(gpu.to_numpy(S))

        assert len(results) == batch_size
        for S in results:
            assert S.shape == (d,)
            assert np.all(S >= 0)  # singular values are non-negative
