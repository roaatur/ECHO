import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam  # new fitting
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.tuner import Tuner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loss functions
def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    """
    Negative Log-Likelihood of Normal-Inverse-Gamma distribution
    From Amini's repo - converted to PyTorch
    """
    twoBlambda = 2 * beta * (1 + v)
    
    nll = (
        0.5 * torch.log(torch.tensor(np.pi) / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    
    return torch.mean(nll) if reduce else nll


def NIG_Reg(y, gamma, v, alpha, beta, reduce=True):
    """
    Evidence Regularizer - penalizes high evidence when prediction is wrong
    """
    error = torch.abs(y - gamma)
    evi = 2 * v + alpha
    reg = error * evi
    
    return torch.mean(reg) if reduce else reg


def EvidentialLoss(y_true, gamma, v, alpha, beta, coeff=1.0):
    """
    Combined loss: NLL + coeff * Regularizer
    """
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    
    return loss_nll + coeff * loss_reg


#the model
class EvidentialLSTM(L.LightningModule):  
    
    def __init__(self, input_size=5, hidden_size=64, horizon=10, lr=1e-3, coeff=0.01):
        super().__init__()   
        self.save_hyperparameters()
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,    
            hidden_size=hidden_size,  
            batch_first=True          
        )
        
        # Output 4 parameters per horizon step: γ, ν, α, β
        self.fc = nn.Linear(hidden_size, horizon * 4)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        hidden = h_n.squeeze(0)
        
        output = self.fc(hidden)
        output = output.view(-1, self.horizon, 4)
        
        # Apply constraints for valid NIG parameters
        gamma = output[..., 0]                      # prediction mean
        v = F.softplus(output[..., 1])              # ν > 0
        alpha = F.softplus(output[..., 2]) + 1      # α > 1
        beta = F.softplus(output[..., 3])           # β > 0
        
        return gamma, v, alpha, beta
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        gamma, v, alpha, beta = self(x)
        loss = EvidentialLoss(y, gamma, v, alpha, beta, coeff=self.hparams.coeff)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        gamma, v, alpha, beta = self(x)
        loss = EvidentialLoss(y, gamma, v, alpha, beta, coeff=self.hparams.coeff)
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
    
    def get_uncertainty(self, v, alpha, beta):
        """Extract uncertainties from NIG parameters (Eq. 5 from Amini)"""
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (v * (alpha - 1))
        return aleatoric, epistemic


def generate_realistic_synthetic_data(
    num_samples=1000,
    T_context=20,
    T_horizon=10,
    input_dim=5,
    noise_level=0.1
):
    X = np.zeros((num_samples, T_context, input_dim), dtype=np.float32)
    y = np.zeros((num_samples, T_horizon), dtype=np.float32)

    for i in range(num_samples):
        t_all = np.arange(T_context + T_horizon)

        freq1 = np.random.uniform(0.1, 0.2)
        freq2 = np.random.uniform(0.2, 0.4)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)

        base_signal = (0.3 * np.sin(freq1 * t_all + phase1) +
                       0.2 * np.sin(freq2 * t_all + phase2))

        if np.random.random() > 0.7:
            jump_time = np.random.randint(10, 25)
            base_signal[jump_time:] += np.random.uniform(-0.2, 0.2)

        complexity = 0.5 + base_signal
        complexity = np.clip(complexity, 0.05, 0.95)

        for f in range(input_dim):
            feature_weight = np.random.uniform(0.3, 0.7)
            independent_signal = 0.3 * np.sin(np.random.uniform(0.1, 0.3) * t_all + np.random.uniform(0, 2*np.pi))
            X[i, :, f] = (feature_weight * complexity[:T_context] +
                         (1 - feature_weight) * independent_signal[:T_context] +
                         np.random.normal(0, noise_level, T_context))

        y[i, :] = complexity[T_context:]

    return X, y


# generate and prepare data
X, y = generate_realistic_synthetic_data(num_samples=2000)
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train
model = EvidentialLSTM(coeff=0.01)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)



#evaluation
model.eval()

with torch.no_grad():
    X_test = torch.from_numpy(X_val[:5])
    y_test = y_val[:5]
    
    gamma, v, alpha, beta = model(X_test)
    predictions = gamma.numpy()
    
    aleatoric, epistemic = model.get_uncertainty(v, alpha, beta)
    aleatoric = aleatoric.numpy()
    epistemic = epistemic.numpy()

# Plot
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Top: predictions with uncertainty band
    axes[0, i].plot(y_test[i], 'b-o', label='actual')
    axes[0, i].plot(predictions[i], 'r-x', label='predicted')
    axes[0, i].fill_between(
        range(10), 
        predictions[i] - np.sqrt(epistemic[i]), 
        predictions[i] + np.sqrt(epistemic[i]), 
        alpha=0.3, color='orange', label='±σ epistemic'
    )
    axes[0, i].set_title(f'Sample {i+1}')
    axes[0, i].set_ylim(0, 1)
    if i == 0:
        axes[0, i].legend(fontsize=7)
    
    # Bottom: uncertainty breakdown
    axes[1, i].plot(aleatoric[i], 's-', label='aleatoric')
    axes[1, i].plot(epistemic[i], '^-', label='epistemic')
    axes[1, i].set_xlabel('Horizon')
    if i == 0:
        axes[1, i].set_ylabel('Uncertainty')
        axes[1, i].legend(fontsize=7)

plt.tight_layout()
plt.show()