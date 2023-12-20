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

def MDN_loss_function(
    targets: torch.tensor, logpi: torch.tensor, mu: torch.tensor, sigma: torch.tensor, beta: int = 0
) -> float:
    targets = targets.unsqueeze(2)
    post_z = Independent(Normal(mu, sigma), 1)

    logprobs = post_z.log_prob(targets)
    weighted_logprobs = logpi + logprobs 

    max_log_probs = torch.max(weighted_logprobs, dim=-1, keepdim=True)[0]
    weighted_logprobs = weighted_logprobs - max_log_probs

    probs = torch.exp(weighted_logprobs)
    probs = torch.sum(probs, dim=-1)

    logprobs = max_log_probs.squeeze() + torch.log(probs)

    nll_loss = -torch.mean(logprobs)

    prior_z = Independent(
        Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1
    )  # ~ Normal(0,1)
    kl_div_loss = kl_divergence(post_z, prior_z).mean()

    total_loss = nll_loss + beta * kl_div_loss

    return total_loss


def sample_gmm(pi: torch.tensor, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
    """
    Sample from a gaussian mixture model

        Parameters:
            pi (n_gaussians): mixing coefficents
            mu (n_gaussians, gaussian_dim): means
            sigma (n_gaussians, gaussian_dim): standard deviations
    """
    categorical = Categorical(probs=pi)
    gaussians = Independent(Normal(loc=mu, scale=sigma), 1)
    mixture_dist = MixtureSameFamily(categorical, gaussians)
    sample = mixture_dist.sample()
    trace = torch.sum(mixture_dist.variance, -1)
    return sample, trace


if __name__ == "__main__":
    softmax = torch.nn.Softmax(dim=-1)

    mu = torch.rand((2, 10))
    sigma = torch.rand((2, 10))
    pi = softmax(torch.rand(2))

    sample = sample_gmm(pi, mu, sigma)
