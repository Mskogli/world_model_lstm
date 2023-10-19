import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List
from gmm import gaussian_ll_loss, KL_divergence_loss


class GMMRNN(nn.Module):
    """
    LSTM based world model which predicts the distribution over the next latent. The hidden state h serves as a compact
    representation of the world, which can be used for control purposes
    """

    def __init__(
        self,
        input_dim=128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_gaussians=5,
        device: str = "cuda:0",
    ) -> None:
        super(GMMRNN, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, n_gaussians)
        self.fc2 = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.fc3 = nn.Linear(hidden_dim, n_gaussians * latent_dim)

        self.device = torch.device(device)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:  # (3,2)-tuple
        y, hidden = self.lstm(x, h)
        pi, mu, sigma = self.get_guassian_coeffs(y)
        return (pi, mu, sigma), hidden

    def get_guassian_coeffs(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:  # 3-tuple
        sequence_length = y.size(1)

        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1, sequence_length, self.n_gaussians)
        mu = mu.view(-1, sequence_length, self.n_gaussians, self.latent_dim)
        sigma = sigma.view(-1, sequence_length, self.n_gaussians, self.latent_dim)

        sigma = torch.exp(sigma)  # Ensure valid values for the normal dist
        logpi = torch.nn.functional.log_softmax(pi, -1)  # Numerical stability

        return logpi, mu, sigma

    def init_hidden_state(self, batch_size: int) -> Tuple[torch.tensor, ...]:  # 2-tuple
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
        )

    def loss_criterion(
        self,
        targets: torch.tensor,
        logpi: torch.tensor,
        mu: torch.tensor,
        sigma: torch.tensor,
    ) -> float:
        # beta_kl = 0.1  # weight of the KL divergence loss term

        loglik_loss = gaussian_ll_loss(targets, logpi, mu, sigma)
        # kl_loss = KL_divergence_loss(logpi, mu, sigma)

        total_loss = loglik_loss
        return total_loss


def detach(states: torch.Tensor) -> List[torch.Tensor]:
    """
    Detach states from the computational graph, truncated backpropagation trough time
    """
    return [state.detach() for state in states]
