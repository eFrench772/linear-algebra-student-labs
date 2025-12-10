import numpy as np
import matplotlib.pyplot as plt
import time


def lu_factorisation(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A, dtype=float), np.zeros_like(A, dtype=float)

    # Set diagonal of L to 1 (unit diagonal)
    for i in range(n):
        L[i, i] = 1
    
    # Compute U's first row
    for j in range(n):
        U[0, j] = A[0, j]
    
    # Compute L's first column (below diagonal)
    for i in range(1, n):
        L[i, 0] = A[i, 0] / U[0, 0]
    
    # Compute remaining elements
    for i in range(1, n):
        # Compute U[i, j] for j >= i (upper triangular part)
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # Compute L[i, j] for j < i (lower triangular part, below diagonal)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U


def forward_solve(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)
    
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
        y[i] /= L[i, i]
    
    return y


def backward_solve(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] /= U[i, i]
    
    return x


def lu_solve(A, b):
    L, U = lu_factorisation(A)
    y = forward_solve(L, b)
    x = backward_solve(U, y)
    return x


def gaussian_elimination(A, b):
    n = len(b)
    # Create augmented matrix
    Ab = np.column_stack([A.copy(), b.copy()])
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Eliminate column entries below pivot
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # Back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - sum(Ab[i, j] * x[j] for j in range(i + 1, n))) / Ab[i, i]
    
    return x


def generate_safe_system(n):
    A = np.random.randn(n, n)
    x = np.random.randn(n)
    b = A @ x
    return A, b, x


if __name__ == "__main__":
    # First, compute the determinant of A_large
    print("Computing determinant of A_large...")
    A_large, b_large, x_large = generate_safe_system(100)
    det_A_large = determinant(A_large)
    print(f"Determinant of A_large (100x100): {det_A_large}")
    print(f"NumPy's determinant (verification): {np.linalg.det(A_large)}")
    print("\n" + "="*60 + "\n")
    
    # Then run the timing comparisons
    sizes = [2**j for j in range(1, 9)]  # Extended range for better comparison
    
    lu_times = []
    ge_times = []
    
    print("Running timing comparisons...")
    print(f"{'Size':<10} {'LU Time (s)':<15} {'GE Time (s)':<15}")
    print("-" * 40)
    
    for n in sizes:
        # Generate a random system of linear equations of size n
        A, b, x_true = generate_safe_system(n)
        
        start = time.time()
        x_lu = lu_solve(A, b)
        lu_time = time.time() - start
        lu_times.append(lu_time)
        
        start = time.time()
        x_ge = gaussian_elimination(A, b)
        ge_time = time.time() - start
        ge_times.append(ge_time)
        
        print(f"{n:<10} {lu_time:<15.6f} {ge_time:<15.6f}")

        assert np.allclose(x_lu, x_true), f"LU solution incorrect for n={n}"
        assert np.allclose(x_ge, x_true), f"GE solution incorrect for n={n}"
    
    # Create plot 1: Absolute runtimes
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, lu_times, 'o-', label='LU Factorisation', linewidth=2, markersize=8)
    plt.plot(sizes, ge_times, 's-', label='Gaussian Elimination', linewidth=2, markersize=8)
    plt.xlabel('Problem Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Runtime Comparison: LU Factorisation vs Gaussian Elimination', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('runtime_comparison_absolute.png', dpi=300, bbox_inches='tight')
    print("\nSaved: runtime_comparison_absolute.png")
    
    plt.show()