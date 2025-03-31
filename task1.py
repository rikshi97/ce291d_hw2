import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

def solve_ks_equation(N=1024, tmax=100, h=0.025, initial_condition=None):
    x = 32*np.pi*np.transpose(np.arange(1, N+1))/N
    if initial_condition is None:
        u = np.cos(x/16)*(1+np.sin(x/16))
    else:
        u = initial_condition
    v = np.fft.fft(u)
    
    h = 0.025
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
    
    uu = np.array([u])
    tt = 0
    nmax = round(tmax/h)
    nplot = int((tmax/250)/h)
    g = -0.5*k
    
    for n in range(1,nmax+1):
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

class KSDataset(Dataset):
    def __init__(self, x, tt, uu):
        self.x = torch.FloatTensor(x)
        self.tt = torch.FloatTensor(tt)
        self.uu = torch.FloatTensor(uu)
        
    def __len__(self):
        return self.uu.shape[0] - 1
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        return self.uu[idx], self.uu[idx + 1]

class KSNetwork(nn.Module):
    def __init__(self, input_size):
        super(KSNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )
        
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (current, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(current)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}')

def plot_comparison(x, tt, exact_solution, predicted_solution, title):
    fig = plt.figure(figsize=(15, 5))
    
    # Exact solution
    ax1 = fig.add_subplot(121, projection='3d')
    tt, x = np.meshgrid(tt, x)
    surf1 = ax1.plot_surface(tt, x, exact_solution.transpose(), cmap=cm.hot)
    ax1.set_title('Exact Solution')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Space')
    ax1.set_zlabel('u(x,t)')
    
    # Neural network prediction
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(tt, x, predicted_solution.transpose(), cmap=cm.hot)
    ax2.set_title('Neural Network Prediction')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Space')
    ax2.set_zlabel('u(x,t)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save with unique filename based on title
    filename = f'ks_comparison_{title.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    
    # Also save the numerical data
    np.save(f'ks_exact_{title.lower().replace(" ", "_")}.npy', exact_solution)
    np.save(f'ks_predicted_{title.lower().replace(" ", "_")}.npy', predicted_solution)
    
    print(f"Saved comparison plot and data for: {title}")

def main():
    # Generate initial solution
    x, tt, uu = solve_ks_equation()
    
    # Plot original solution
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    tt_mesh, x_mesh = np.meshgrid(tt, x)
    surf = ax.plot_surface(tt_mesh, x_mesh, uu.transpose(), cmap=cm.hot, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Original KS Solution')
    plt.savefig('ks_original.png')
    plt.close()
    
    # Save original solution data
    np.save('ks_original_solution.npy', uu)
    np.save('ks_original_x.npy', x)
    np.save('ks_original_t.npy', tt)
    
    # Create dataset and dataloader for training
    dataset = KSDataset(x, tt, uu)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = KSNetwork(input_size=len(x))
    
    # Train model
    print("Training neural network...")
    train_model(model, train_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'ks_model.pth')
    
    # Generate predictions for different initial conditions
    initial_conditions = [
        # Original conditions
        (np.cos(x/8)*(1+np.sin(x/8)), "Different frequency"),
        (np.sin(x/16), "Pure sine"),
        (np.cos(x/16), "Pure cosine"),
        
        # Additional test conditions
        (np.sin(x/4), "Higher frequency sine"),
        (np.cos(x/4), "Higher frequency cosine"),
        (np.sin(x/8) + np.cos(x/8), "Combined sine and cosine"),
        (np.exp(-x**2/100), "Gaussian"),
        (np.tanh(x/8), "Tanh function"),
        (np.sin(x/16) * np.exp(-x**2/200), "Damped sine wave")
    ]
    
    # Create a results directory if it doesn't exist
    os.makedirs('ks_results', exist_ok=True)
    
    for init_cond, title in initial_conditions:
        # Generate exact solution
        x_test, tt_test, exact_solution = solve_ks_equation(initial_condition=init_cond)
        
        # Generate neural network predictions
        model.eval()
        with torch.no_grad():
            predicted_solution = []
            current = torch.FloatTensor(init_cond).unsqueeze(0)
            
            for _ in range(len(tt_test)):
                next_state = model(current)
                predicted_solution.append(next_state.numpy()[0])
                current = next_state
            
            predicted_solution = np.array(predicted_solution)
        
        # Plot comparison
        plot_comparison(x_test, tt_test, exact_solution, predicted_solution, title)
        
        # Save initial condition
        np.save(f'ks_results/initial_condition_{title.lower().replace(" ", "_")}.npy', init_cond)

if __name__ == "__main__":
    main()