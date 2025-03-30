import numpy as np
from scipy import linalg

def check_controllability(A, B):
    """Check if the system is controllable using Kalman's controllability test."""
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(C)
    return rank == n

def check_observability(A, C):
    """Check if the system is observable using Kalman's observability test."""
    n = A.shape[0]
    O = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    rank = np.linalg.matrix_rank(O)
    return rank == n

def place_poles(A, B, desired_poles):
    """Place poles using Ackermann's formula."""
    n = A.shape[0]
    
    # Compute controllability matrix
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    
    # Compute characteristic polynomial coefficients
    char_poly = np.poly(desired_poles)
    
    # Compute Ackermann's formula
    K = np.zeros((1, n))
    for i in range(n):
        K += char_poly[i] * np.linalg.matrix_power(A, n-1-i)
    
    # Compute feedback gain
    K = K @ np.linalg.inv(C)
    return K

def main():
    # Task 4.1: Check controllability and observability
    print("Task 4.1: Controllability and Observability Analysis")
    print("-" * 50)
    
    # System matrices
    A = np.array([[0, 1, 0],
                  [0, 1, 1],
                  [-1, -2, -3]])
    B = np.array([[1],
                  [0],
                  [0]])
    C = np.array([[1, 0, 1]])
    
    # Check controllability
    is_controllable = check_controllability(A, B)
    print(f"System is {'controllable' if is_controllable else 'not controllable'}")
    
    # Check observability
    is_observable = check_observability(A, C)
    print(f"System is {'observable' if is_observable else 'not observable'}")
    
    # Task 4.2: Design stabilizing controller
    print("\nTask 4.2: Stabilizing Controller Design")
    print("-" * 50)
    
    # System matrices for the second system
    A2 = np.array([[1, 1],
                   [1, 2]])
    B2 = np.array([[1],
                   [0]])
    
    # Design controller
    desired_poles = [-1, -2]
    K = place_poles(A2, B2, desired_poles)
    print("Feedback gain matrix K:")
    print(K)
    
    # Verify stability
    A_closed = A2 - B2 @ K
    eigenvalues = np.linalg.eigvals(A_closed)
    print("\nClosed-loop eigenvalues:")
    print(eigenvalues)
    print(f"System is {'stable' if all(np.real(eigenvalues) < 0) else 'unstable'}")

if __name__ == "__main__":
    main() 