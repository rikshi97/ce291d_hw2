import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Neural Network Architecture for time evolution
class KSNet(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512):
        super(KSNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        return self.network(x)

def solve_ks_exact(initial_condition, N=1024, tmax=100, h=0.025):
    # Use the provided initial condition
    x = 32*np.pi*np.transpose(np.arange(1, N+1))/N
    u = initial_condition
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

    # Initialization
    uu = np.array([u])
    tt = np.array([0])
    nmax = round(tmax/h)
    
    # Ensure nplot is not zero by checking tmax
    if tmax >= 250*h:
        nplot = int((tmax/250)/h)
    else:
        # If tmax is small, save every 5 steps
        nplot = 5
        
    g = -0.5*k

    # Time stepping loop
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
            tt = np.append(tt, t)
    
    return uu, tt, x

def generate_initial_conditions(N=1024):
    x = 32*np.pi*np.transpose(np.arange(1, N+1))/N
    # Three different initial conditions with descriptions
    ic1 = np.cos(x/16)*(1+np.sin(x/16))  # Original
    ic2 = np.sin(x/8) + 0.5*np.cos(x/4)  # Different frequency
    ic3 = np.exp(-(x-16*np.pi)**2/100)    # Gaussian
    
    # Return conditions, descriptions, and x values
    conditions = [ic1, ic2, ic3]
    descriptions = ["Original: cos(x/16)(1+sin(x/16))", 
                   "Different Frequency: sin(x/8)+0.5cos(x/4)", 
                   "Gaussian: exp(-(x-16π)²/100)"]
    return conditions, descriptions, x

def generate_training_data(initial_condition, num_steps=5):
    """Generate input-output pairs for time evolution"""
    # Ensure we have a minimum tmax value that will generate enough steps
    tmax = max(num_steps*0.025, 0.5)  # At least 0.5 to ensure we get enough timesteps
    uu, _, _ = solve_ks_exact(initial_condition, tmax=tmax, h=0.025)
    input_data = []
    output_data = []
    
    for i in range(len(uu)-1):
        input_data.append(uu[i])
        output_data.append(uu[i+1])
    
    return torch.FloatTensor(input_data), torch.FloatTensor(output_data)

def train_network(model, inputs, targets, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def nn_solve(model, initial_condition, num_steps, h=0.025):
    """Use trained neural network to solve KS equation for multiple steps"""
    u = torch.FloatTensor(initial_condition)
    uu = [initial_condition.copy()]
    tt = [0]
    
    # Simulate forward in time using the neural network
    for i in range(num_steps):
        t = (i+1) * h
        with torch.no_grad():
            u = model(u)
        uu.append(u.numpy())
        tt.append(t)
    
    return np.array(uu), np.array(tt)

def plot_comparison(exact_sol, nn_sol, tt_exact, tt_nn, x, title, condition_type):
    fig = plt.figure(figsize=(12, 5))
    
    # Create meshgrids for plotting
    tt_exact_grid, x_grid = np.meshgrid(tt_exact, x)
    tt_nn_grid, x_grid_nn = np.meshgrid(tt_nn, x)
    
    # Exact solution
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(tt_exact_grid, x_grid, exact_sol.transpose(), cmap=cm.hot, linewidth=0, antialiased=False)
    ax1.set_title('Exact Solution')
    fig.colorbar(surf1, shrink=0.5, aspect=5)
    
    # Neural network solution
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(tt_nn_grid, x_grid_nn, np.array(nn_sol).transpose(), cmap=cm.hot, linewidth=0, antialiased=False)
    ax2.set_title('Neural Network Solution')
    fig.colorbar(surf2, shrink=0.5, aspect=5)
    
    plt.suptitle(f"{title}: {condition_type}")
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

def main():
    # Generate initial conditions
    N = 1024
    tmax = 10  # Reduced for faster computation
    h = 0.025
    num_timesteps = int(tmax / h)
    
    initial_conditions, descriptions, x = generate_initial_conditions(N)
    
    # Create and train neural network
    model = KSNet()
    
    # Generate training data from the first initial condition
    train_inputs, train_targets = generate_training_data(initial_conditions[0], num_steps=50)
    
    # Train the model
    print("Training neural network...")
    train_network(model, train_inputs, train_targets, epochs=100)
    
    # Generate predictions for all initial conditions
    print("\nGenerating predictions for different initial conditions...")
    for i, (ic, desc) in enumerate(zip(initial_conditions, descriptions)):
        print(f"Processing initial condition {i+1}...")
        
        # Get exact solution
        exact_sol, tt_exact, _ = solve_ks_exact(ic, N, tmax, h)
        
        # Generate neural network solution
        nn_sol, tt_nn = nn_solve(model, ic, num_timesteps)
        
        # Plot comparison
        plot_comparison(exact_sol, nn_sol, tt_exact, tt_nn, x, f'Initial Condition {i+1}', desc)

if __name__ == "__main__":
    main() 