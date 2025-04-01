import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

def main():
    # Problem 1: Determine controllability and observability
    print("Problem 1: System Controllability and Observability Analysis")
    
    # Define system matrices
    A = np.array([[0, 1, 0],
                  [0, 1, 1],
                  [-1, -2, -3]])
    
    B = np.array([[1],
                  [0],
                  [0]])
    
    C = np.array([[1, 0, 1]])
    
    D = np.array([[1]])
    
    # Create state-space system
    sys = ctrl.StateSpace(A, B, C, D)
    
    # Check controllability
    Co = ctrl.ctrb(A, B)
    rank_Co = np.linalg.matrix_rank(Co)
    n = A.shape[0]  # System order
    
    print(f"Controllability matrix:\n{Co}")
    print(f"Rank of controllability matrix: {rank_Co}")
    print(f"System order: {n}")
    print(f"Is the system controllable? {rank_Co == n}")
    
    # Check observability
    Ob = ctrl.obsv(A, C)
    rank_Ob = np.linalg.matrix_rank(Ob)
    
    print(f"\nObservability matrix:\n{Ob}")
    print(f"Rank of observability matrix: {rank_Ob}")
    print(f"Is the system observable? {rank_Ob == n}")
    
    # Problem 2: Find stabilizing feedback control
    print("\nProblem 2: Stabilizing Feedback Control Design")
    
    # Define system matrices
    A2 = np.array([[1, 1],
                   [1, 2]])
    
    B2 = np.array([[1],
                   [0]])
    
    # Check system eigenvalues (before control)
    eig_A2 = np.linalg.eigvals(A2)
    print(f"Original system eigenvalues: {eig_A2}")
    print(f"Is the original system stable? {np.all(np.real(eig_A2) < 0)}")
    
    # Check controllability
    Co2 = ctrl.ctrb(A2, B2)
    rank_Co2 = np.linalg.matrix_rank(Co2)
    n2 = A2.shape[0]
    
    print(f"Controllability matrix:\n{Co2}")
    print(f"Rank of controllability matrix: {rank_Co2}")
    print(f"Is the system controllable? {rank_Co2 == n2}")
    
    if rank_Co2 == n2:
        # Choose desired eigenvalues in the left half-plane
        desired_eigs = [-1, -2]  # Example stable poles
        
        # Compute feedback gain K using pole placement
        K = ctrl.place(A2, B2, desired_eigs)
        
        print(f"\nFeedback gain K = {K}")
        
        # Verify closed-loop system stability
        A_cl = A2 - B2 @ K
        eig_cl = np.linalg.eigvals(A_cl)
        
        print(f"Closed-loop eigenvalues: {eig_cl}")
        print(f"Is the closed-loop system stable? {np.all(np.real(eig_cl) < 0)}")
    else:
        print("System is not controllable, cannot place poles arbitrarily.")

if __name__ == "__main__":
    main()
