import pandas as pd
import numpy as np
from scipy import linalg

# Load snapshot matrices
X1 = pd.read_csv('hw2-DMD-X0.csv', header=None)
X1 = X1.applymap(lambda s: complex(s.replace('i', 'j'))).values # This will give you the first snapshot matrix X
X2 = pd.read_csv('hw2-DMD-X1.csv', header=None)
X2 = X2.applymap(lambda s: complex(s.replace('i', 'j'))).values # This will give you the second snapshot matrix X'

# Compute SVD of X1
U, S, Vh = linalg.svd(X1)

# Truncate to r modes (using 90% energy criterion)
energy = np.cumsum(S) / np.sum(S)
r = np.argmax(energy >= 0.9) + 1
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vhr = Vh[:r, :]

# Compute DMD matrix
Atilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.linalg.inv(Sr)

# Compute eigenvalues and eigenvectors of Atilde
eigvals, eigvecs = linalg.eig(Atilde)

# Compute DMD modes
Phi = X2 @ Vhr.conj().T @ np.linalg.inv(Sr) @ eigvecs

# Sort eigenvalues by magnitude
idx = np.argsort(np.abs(eigvals))[::-1]
eigvals = eigvals[idx]
Phi = Phi[:, idx]

# Save results
np.save('dmd_eigenvalues.npy', eigvals)
np.save('dmd_modes.npy', Phi)

print(f"Number of DMD modes: {r}")
print(f"Eigenvalues shape: {eigvals.shape}")
print(f"DMD modes shape: {Phi.shape}")


