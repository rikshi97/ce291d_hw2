import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class KSDataset(Dataset):
    def __init__(self, t, x, u):
        # Normalize the data
        self.u_mean = np.mean(u)
        self.u_std = np.std(u)
        u_normalized = (u - self.u_mean) / (self.u_std + 1e-8)
        
        self.t = torch.FloatTensor(t)
        self.x = torch.FloatTensor(x)
        self.u = torch.FloatTensor(u_normalized)
        
    def __len__(self):
        return len(self.t) - 1
        
    def __getitem__(self, idx):
        # Return current state and next state
        current = self.u[idx]
        next_state = self.u[idx + 1]
        return current, next_state
    
    def denormalize(self, u_normalized):
        return u_normalized * (self.u_std + 1e-8) + self.u_mean

class KSNetwork(nn.Module):
    def __init__(self, input_size):
        super(KSNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, input_size)
        )
        
        # Initialize weights with small values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.net(x)

def solve_ks(t, x, u0, dt, dx):
    """Solve KS equation using upwind scheme for stability"""
    N = len(x)
    u = np.zeros((len(t), N))
    u[0] = u0
    
    # Add padding for periodic boundary conditions
    def pad_periodic(arr):
        return np.pad(arr, 2, mode='wrap')
    
    # Compute derivatives with upwind scheme
    def derivatives(u):
        u_pad = pad_periodic(u)
        
        # First derivative (upwind)
        ux = np.zeros_like(u)
        ux[u > 0] = (u_pad[2:-2][u > 0] - u_pad[1:-3][u > 0]) / dx
        ux[u <= 0] = (u_pad[3:-1][u <= 0] - u_pad[2:-2][u <= 0]) / dx
        
        # Second derivative (central)
        uxx = (u_pad[3:-1] - 2*u_pad[2:-2] + u_pad[1:-3]) / (dx*dx)
        
        # Fourth derivative (central with artificial dissipation)
        uxxxx = (u_pad[4:] - 4*u_pad[3:-1] + 6*u_pad[2:-2] - 
                4*u_pad[1:-3] + u_pad[0:-4]) / (dx**4)
        uxxxx += 1e-4 * (np.roll(u, 1) - 2*u + np.roll(u, -1)) / (dx*dx)
        
        return ux, uxx, uxxxx
    
    # Time stepping with adaptive timestep
    dt_base = dt
    for n in range(len(t)-1):
        # Compute local CFL number and adjust timestep
        max_speed = np.max(np.abs(u[n]))
        dt = min(dt_base, 0.5 * dx / (max_speed + 1e-8))
        
        # RK2 time stepping
        ux, uxx, uxxxx = derivatives(u[n])
        k1 = -u[n]*ux - uxx - uxxxx
        
        u_mid = u[n] + 0.5*dt*k1
        ux, uxx, uxxxx = derivatives(u_mid)
        k2 = -u_mid*ux - uxx - uxxxx
        
        u[n+1] = u[n] + dt*k2
        
        # Add small amount of filtering
        u[n+1] = 0.99*u[n+1] + 0.01*(np.roll(u[n+1], 1) + np.roll(u[n+1], -1))/2
    
    return u

def generate_data(t, x, u0, dt, dx):
    u = solve_ks(t, x, u0, dt, dx)
    return t, x, u

def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (current, target) in enumerate(train_loader):
            current = current.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(current)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_ks_model.pth')
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

def main():
    # Parameters
    L = 32
    N = 64  # Further reduced spatial resolution for stability
    dt = 0.1
    t = np.arange(0, 20, dt)  # Reduced time span for stability
    dx = L/N
    x = np.arange(0, L, dx)
    
    # Initial condition (very simple for stability)
    u0 = 0.1*np.sin(2*np.pi*x/L)
    
    # Generate data
    t, x, u = generate_data(t, x, u0, dt, dx)
    
    # Create dataset and dataloader
    dataset = KSDataset(t, x, u)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = KSNetwork(input_size=N).to(device)
    
    # Train model
    train_model(model, train_loader, num_epochs=100, learning_rate=0.001)
    
    # Load best model
    checkpoint = torch.load('best_ks_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = []
        current_state = torch.FloatTensor(dataset.u[0]).unsqueeze(0).to(device)
        
        for _ in range(len(t)):
            next_state = model(current_state)
            predictions.append(next_state.cpu().numpy()[0])
            current_state = next_state
        
        predictions = np.array(predictions)
        
        # Denormalize predictions
        predictions = dataset.denormalize(predictions)
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    
    # Original solution
    ax1 = fig.add_subplot(121, projection='3d')
    tt, xx = np.meshgrid(t, x)
    surf1 = ax1.plot_surface(tt, xx, u.T, cmap='viridis')
    ax1.set_title('Original KS Solution')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('u')
    
    # Neural network predictions
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(tt, xx, predictions.T, cmap='viridis')
    ax2.set_title('Neural Network Predictions')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u')
    
    plt.tight_layout()
    plt.savefig('ks_neural_net_prediction.png')
    plt.close()

if __name__ == "__main__":
    main()