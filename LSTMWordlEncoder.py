import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List

class LSTMWorldModel(nn.Module):
    """
    LSTM based world model which predicts the distribution over the next latent. The hidden state h serves as a compact
    representation of the world, which can be used for control purposes
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super(LSTMWorldModel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(latent_dim, hidden_dim, 1, batch_first = True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]] : # (2,2)-tuple

        y, (h, c) = self.lstm(x, h)
        mu, sigma = self.get_guassian_coeffs(y)

        return (mu, sigma), (h, c)
    
    def get_guassian_coeffs(self, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # 2-tuple
        rollout_length = y.size(1)

        mu, sigma = self.fc_mu(y), self.fc_sigma(y)
        mu = mu.view(-1, rollout_length, self.latent_dim)
        sigma = sigma.view(-1, rollout_length, self.latent_dim)
        sigma = torch.exp(sigma)
    
        return mu, sigma
    
def KLregularizedLogLikelihoodLoss(targets: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the log-likelihood loss of the measured latent vectors with a regularizing KL-divergence term
    which makes the predicted distribution consistent with the prioer over the latent variables which is used in the
    seVAE loss.
    """
    latent_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    log_likelihood_loss = latent_distribution.log_prob(targets)
    log_likelihood_loss = -log_likelihood_loss.mean()

    beta = 0.1
    standard_normal_distribution = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(sigma))
    kl_divergence_loss = torch.distributions.kl.kl_divergence(latent_distribution, standard_normal_distribution)
    kl_divergence_loss = kl_divergence_loss.sum() / targets.size(0)# Not mathematically correct

    total_loss = log_likelihood_loss + beta*kl_divergence_loss
    return total_loss

def detach(states: torch.Tensor) -> List[torch.Tensor]:
    """
    Detach states from the computational graph in order to facilitate truncated backprop trough time
    """
    return [state.detach() for state in states] 