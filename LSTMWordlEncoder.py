import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List

class LSTMWorldModel(nn.Module):
    """
    LSTM based world model which predicts the distribution over the next latent. The hidden state h serves as a compact
    representation of the world, which can be used for control purposes
    """
    def __init__(self, input_dim = 128, latent_dim: int = 128, hidden_dim: int = 256, n_gaussians = 5) -> None:
        super(LSTMWorldModel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first = True)
        self.fc1 = nn.Linear(hidden_dim, n_gaussians)
        self.fc2 = nn.Linear(hidden_dim, n_gaussians*latent_dim)
        self.fc3 = nn.Linear(hidden_dim, n_gaussians*latent_dim)


    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]] : # (2,2)-tuple

        y, hidden = self.lstm(x, h)
        pi, mu, sigma = self.get_guassian_coeffs(y)
        return (pi, mu, sigma), hidden
    
    def get_guassian_coeffs(self, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # 2-tuple
        rollout_length = y.size(1)

        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        pi = pi.view(-1, rollout_length, self.n_gaussians)        
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.latent_dim)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.latent_dim)

        sigma = torch.exp(sigma) # Ensure valid values for the normal dist
        pi = torch.nn.functional.log_softmax(pi, -1)
    
        return pi, mu, sigma
    
def KLregularizedLogLikelihoodLoss(targets: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the log-likelihood loss of the measured latent vectors with a regularizing KL-divergence term
    which makes the predicted distribution consistent with the prioer over the latent variables which is used in the
    seVAE loss.
    """
    latent_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    log_likelihood_loss = -latent_distribution.log_prob(targets)
    log_likelihood_loss = log_likelihood_loss.mean()

    beta = 0.1
    standard_normal_distribution = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(sigma))
    kl_divergence_loss = torch.distributions.kl.kl_divergence(latent_distribution, standard_normal_distribution)
    kl_divergence_loss = kl_divergence_loss.sum() / targets.size(0) 

    total_loss = log_likelihood_loss + beta*kl_divergence_loss
    return total_loss

def detach(states: torch.Tensor) -> List[torch.Tensor]:
    """
    Detach states from the computational graph in order to facilitate truncated backprop trough time
    """
    return [state.detach() for state in states]

def mdn_loss_fn(y, logpi, mu, sigma):
    z_score = (y.unsqueeze(2) - mu) / sigma
    normal_loglik = (
        -1/2*torch.einsum("bsdc, bsdc ->bsd", z_score, z_score)
        -torch.sum(torch.log(sigma), dim=-1)
    )
    loglik = torch.logsumexp(logpi + normal_loglik, dim=-1)
    loglik = torch.sum(loglik, dim=-1)
    loglik = loglik.mean()
    return -loglik


def criterion(y, pi, mu, sigma):
    return mdn_loss_fn(y, pi, mu, sigma)