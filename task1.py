import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the neural network
class KSNetwork(nn.Module):
    def __init__(self):
        super(KSNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Function to generate exact solution (from the original code)
def generate_exact_solution(N=1024, tmax=100, h=0.025):
    # Initial condition
    x = 32*np.pi*np.transpose(np.arange(1, N+1))/N
    u = np.cos(x/16)*(1+np.sin(x/16))
    v = np.fft.fft(u)
    
    # Numerical grid
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16
    L = np.power(k,2) - np.power(k,4)
    E = np.exp(h*L)
    E2 = np.exp(h*L/2)
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    # Time stepping
    uu = np.array([u])
    tt = 0
    nmax = round(tmax/h)
    nplot = int((tmax/250)/h)
    g = -0.5*k
    
    for n in range(1, nmax+1):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.power(np.fft.ifft(v),2)))
        a = E2*v+Q*Nv
        Na = g*np.fft.fft(np.real(np.power(np.fft.ifft(a),2)))
        b = E2*v +Q*Na
        Nb = g*np.fft.fft(np.real(np.power(np.fft.ifft(b),2)))
        c = E2*a+Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.power(np.fft.ifft(c),2)))
        v = E*v + Nv*f1+2*(Na+Nb)*f2+Nc*f3
        if n%nplot == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
    
    return x, tt, uu

# Generate training data
N = 1024
x, tt, uu_exact = generate_exact_solution(N=N)

# Create a grid of x and tt
x_grid, tt_grid = np.meshgrid(x, tt)
X = np.column_stack((x_grid.flatten(), tt_grid.flatten()))
y = uu_exact.flatten()

# Normalize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# Convert to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)  # Reshape to (N, 1)

# Create and train the model
model = KSNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2000  # Increased epochs
batch_size = 128   # Increased batch size
for epoch in range(num_epochs):
    # Generate random batch indices
    indices = torch.randperm(len(X))[:batch_size]
    X_batch = X[indices]
    y_batch = y[indices]
    
    # Forward pass
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate predictions
model.eval()
with torch.no_grad():
    X_pred = torch.FloatTensor(X)  # Use the already normalized X
    y_pred = model(X_pred).numpy()
    y_pred = y_pred * y_std + y_mean  # Denormalize
    y_pred = y_pred.reshape(uu_exact.shape)

# Plot comparison
fig = plt.figure(figsize=(15, 5))

# Plot exact solution
ax1 = fig.add_subplot(121, projection='3d')
x_mesh, tt_mesh = np.meshgrid(x, tt)
surf1 = ax1.plot_surface(x_mesh, tt_mesh, uu_exact, cmap=cm.hot, linewidth=0, antialiased=False)
ax1.set_title('Exact Solution')
fig.colorbar(surf1, shrink=0.5, aspect=5)

# Plot neural network prediction
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(x_mesh, tt_mesh, y_pred, cmap=cm.hot, linewidth=0, antialiased=False)
ax2.set_title('Neural Network Prediction')
fig.colorbar(surf2, shrink=0.5, aspect=5)

plt.tight_layout()
# Save the plots
plt.savefig('ks_comparison.png')
# plt.show()  # Comment out the display line

# Calculate and print error
error = np.mean(np.abs(uu_exact - y_pred))
print(f'Average absolute error: {error:.4f}')
