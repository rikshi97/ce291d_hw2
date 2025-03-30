import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

def load_data():
    # Load state and derivative data
    X = pd.read_csv('hw2-ID-X.csv', header=None).values
    Xprime = pd.read_csv('hw2-ID-Xprime.csv', header=None).values
    return X, Xprime

def create_library(X):
    # Create polynomial library up to 3rd order
    x1, x2 = X[:, 0], X[:, 1]
    library = np.column_stack([
        np.ones_like(x1),  # constant term
        x1, x2,           # linear terms
        x1**2, x2**2, x1*x2,  # quadratic terms
        x1**3, x2**3, x1**2*x2, x1*x2**2  # cubic terms
    ])
    return library

def sparse_regression(library, Xprime, threshold=0.1):
    # Solve the sparse regression problem
    Xi = linalg.lstsq(library, Xprime)[0]
    
    # Threshold small coefficients
    Xi[np.abs(Xi) < threshold] = 0
    
    return Xi

def plot_results(X, Xprime, Xi, library):
    # Plot original trajectories
    plt.figure(figsize=(12, 4))
    
    # Plot state variables
    plt.subplot(121)
    plt.plot(X[:, 0], X[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('State Trajectory')
    
    # Plot derivatives
    plt.subplot(122)
    plt.plot(Xprime[:, 0], Xprime[:, 1])
    plt.xlabel('dx1/dt')
    plt.ylabel('dx2/dt')
    plt.title('Derivative Trajectory')
    
    plt.tight_layout()
    plt.savefig('system_identification.png')
    plt.close()
    
    # Print identified equations
    print("\nIdentified Equations:")
    print("dx1/dt =")
    terms = ['1', 'x1', 'x2', 'x1^2', 'x2^2', 'x1*x2', 'x1^3', 'x2^3', 'x1^2*x2', 'x1*x2^2']
    for i, term in enumerate(terms):
        if abs(Xi[0, i]) > 0.1:
            print(f"{Xi[0, i]:.3f}*{term} +")
    
    print("\ndx2/dt =")
    for i, term in enumerate(terms):
        if abs(Xi[1, i]) > 0.1:
            print(f"{Xi[1, i]:.3f}*{term} +")

def main():
    # Load data
    X, Xprime = load_data()
    
    # Create library of candidate functions
    library = create_library(X)
    
    # Perform sparse regression
    Xi = sparse_regression(library, Xprime)
    
    # Plot and print results
    plot_results(X, Xprime, Xi, library)

if __name__ == "__main__":
    main() 