import torch
import numpy as np

from torch.distributions import (
    Normal,
    Categorical,
    MixtureSameFamily,
    Independent,
    kl_divergence,
)


def logavgexp(x: torch.tensor, dim: int) -> torch.tensor:
    if x.size(dim) > 1:
        # TODO: cast to float32 here for IWAE?
        return x.logsumexp(dim=dim) - np.log(x.size(dim))
    else:
        return x.squeeze(dim)


def gaussian_ll_loss(
    targets: torch.tensor, logpi: torch.tensor, mu: torch.tensor, sigma: torch.tensor
) -> float:
    """
    Computes the negative log likelihood of the targets under the predicted distribution of the latent space:
    NLL = -logsumexp_k(log(pi_k) - 1/2(y - u)^T*Sigma^-1*(y - u) - 1/2log(det(Sigma)))

        Parameters:
            targets (torch.tensor): latent targets (batch, seq_l, latent_dim)
            logpi (torch.tensor): log mixing coeffs (batch, seq_l, n_gaussians)
            mu (torch.tensor): predicted means (batch, seq_l, n_gaussians, latent_dim)
            sigma (torch.tensor): predicted standard deviations (batch, seq_l, n_gaussians, latent_dim)
    """
    z_score = (
        targets.unsqueeze(
            2
        )  # (batch, seq_l, latent_dim) -> (batch, seq_l, 1, latent_dim)
        - mu  # (batch, seq_l, num_gaussians, latent_dim)
    ) / sigma  # (batch, seq_l, num_gaussians, latent_dim)

    normal_loglik = -1 / 2 * torch.einsum(
        "bsdc, bsdc ->bsd", z_score, z_score
    ) - torch.sum(torch.log(sigma), dim=-1)

    loglik = torch.logsumexp(
        logpi + normal_loglik, dim=2
    )  # GMM LL normalized by sequence length

    loglik = loglik.mean()

    return -loglik


def MDN_loss_function(
    targets: torch.tensor, logpi: torch.tensor, mu: torch.tensor, sigma: torch.tensor
) -> float:
    targets = targets.unsqueeze(2)
    post_z = Independent(Normal(mu, sigma), 1)  # Posteriror Z dist = Diagnonal Normal

    logprobs = post_z.log_prob(targets)
    weighted_logprobs = logpi + logprobs  # Sum (-1)

    max_log_probs = torch.max(weighted_logprobs, dim=-1, keepdim=True)[0]
    weighted_logprobs = weighted_logprobs - max_log_probs

    probs = torch.exp(weighted_logprobs)
    probs = torch.sum(probs, dim=-1)

    logprobs = max_log_probs.squeeze() + torch.log(probs)

    nll_loss = -torch.mean(logprobs)

    beta = 0
    prior_z = Independent(
        Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1
    )  # ~ Normal(0,1)
    kl_div_loss = kl_divergence(post_z, prior_z).mean()
    # kl_div_loss = logavgexp(kl_div_loss, dim=2).mean()

    total_loss = nll_loss + beta * kl_div_loss

    return total_loss


def sample_gmm(pi: torch.tensor, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
    """
    Sample from a gaussian mixture model

        Parameters:
            pi (n_gaussians)
            mu (n_gaussians, gaussian_dim)
            sigma (n_gaussians, gaussian_dim)
    """

    categorical = Categorical(probs=pi)
    gaussians = Independent(Normal(loc=mu, scale=sigma), 1)
    mixture_dist = MixtureSameFamily(categorical, gaussians)

    sample = mixture_dist.sample()
    return sample


if __name__ == "__main__":
    softmax = torch.nn.Softmax(dim=-1)

    mu = torch.rand((2, 10))
    sigma = torch.rand((2, 10))
    pi = softmax(torch.rand(2))

    sample = sample_gmm(pi, mu, sigma)
