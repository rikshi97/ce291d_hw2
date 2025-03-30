import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.cuda.amp import autocast, GradScaler
import multiprocessing as mp
from functools import partial

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class KSDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data).to(device)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        return (self.data[idx:idx+self.sequence_length], 
                self.data[idx+self.sequence_length])

class KSNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(KSNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def parallel_generate_data(N, tmax, h, num_workers=4):
    # Split the domain into chunks for parallel processing
    chunk_size = N // num_workers
    chunks = [(i*chunk_size, (i+1)*chunk_size) for i in range(num_workers)]
    
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(generate_chunk, [(chunk, tmax, h) for chunk in chunks])
    
    # Combine results
    uu_combined = np.concatenate([r[0] for r in results], axis=1)
    x_combined = np.concatenate([r[1] for r in results])
    tt = results[0][2]  # Time grid is the same for all chunks
    
    # Ensure shapes are correct
    uu_combined = uu_combined.reshape(-1, N)  # Reshape to (time_steps, N)
    x_combined = x_combined.reshape(N)  # Reshape to (N,)
    
    return uu_combined, x_combined, tt

def generate_chunk(chunk_range, tmax, h):
    start, end = chunk_range
    N = end - start
    x = 32*np.pi*np.transpose(np.arange(start+1, end+1))/N
    u = np.cos(x/16)*(1+np.sin(x/16))
    v = np.fft.fft(u)
    
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
    
    return uu, x, tt

def train_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    # Generate training data in parallel
    N = 1024
    uu, x, tt = parallel_generate_data(N, tmax=100, h=0.025)
    
    # Create dataset and dataloader with multiple workers
    sequence_length = 10
    dataset = KSDataset(uu, sequence_length)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    # Initialize model and move to GPU
    input_size = uu.shape[1]
    model = KSNetwork(input_size).to(device)
    
    # Train model
    train_model(model, train_loader)
    
    # Save model
    torch.save(model.state_dict(), 'ks_model.pth')
    
    # Visualize results
    model.eval()
    with torch.no_grad():
        input_seq = torch.FloatTensor(uu[:sequence_length]).unsqueeze(0).to(device)
        predictions = []
        
        for _ in range(len(uu) - sequence_length):
            output = model(input_seq)
            predictions.append(output.cpu().numpy())
            input_seq = torch.cat([input_seq[:, 1:], output.unsqueeze(1)], dim=1)
    
    predictions = np.array(predictions).squeeze()
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    
    # Plot original solution
    ax1 = fig.add_subplot(121, projection='3d')
    tt, x = np.meshgrid(tt, x)
    surf1 = ax1.plot_surface(tt, x, uu.transpose(), cmap=cm.hot, linewidth=0, antialiased=False)
    ax1.set_title('Original Solution')
    fig.colorbar(surf1, shrink=0.5, aspect=5)
    
    # Plot predicted solution
    ax2 = fig.add_subplot(122, projection='3d')
    t_pred = tt[0, sequence_length:]
    # Create meshgrid with correct dimensions
    x_subset = np.linspace(x.min(), x.max(), N)  # Create a matching spatial grid
    tt_pred, x_pred = np.meshgrid(t_pred, x_subset)
    # Ensure predictions shape matches the meshgrid
    predictions = predictions.reshape(-1, N)  # Reshape to match spatial dimension
    surf2 = ax2.plot_surface(tt_pred.T, x_pred.T, predictions, cmap=cm.hot, linewidth=0, antialiased=False)
    ax2.set_title('Neural Network Prediction')
    fig.colorbar(surf2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('ks_neural_net_prediction.png')

if __name__ == "__main__":
    main()