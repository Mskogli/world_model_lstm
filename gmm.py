import torch

from torch.distributions import (
    Normal,
    Categorical,
    MixtureSameFamily,
    Independent,
)


def gaussian_ll_loss(
    targets: torch.tensor, logpi: torch.tensor, mu: torch.tensor, sigma: torch.tensor
) -> float:
    """
    Calculates the negative log likelihood of the targets under the predicted distribution of the latent space:
    NLL = -logsumexp_k(log(pi_k) - 1/2(y - u)^T*Sigma^-1*(y - u) - 1/2log(det(Sigma)))

        Parameters:
            targets (batch, seq_l, latent_dim): latent targets
            logpi (batch, seq_l, n_gaussians): log mixing coeffs
            mu (batch, seq_l, n_gaussians, latent_dim): predicted means
            sigma (batch, seq_l, n_gaussians, latent_dim): predicted standard deviations
    """
    z_score = (
        targets.unsqueeze(
            2
        )  # (batch, seq_l, latent_dim) -> (batch, seq_l, 1, latent_dim)
        - mu  
    ) / sigma

    normal_loglik = -1 / 2 * torch.einsum(
        "bsdc, bsdc ->bsd", z_score, z_score
    ) - torch.sum(torch.log(sigma), dim=-1)

    loglik = torch.logsumexp(
        logpi + normal_loglik, dim=2
    ) 
    loglik = loglik.mean()

    return -loglik


def KL_divergence_loss(
    mu: torch.tensor, sigma: torch.tensor
) -> float:
    """
    Calculates the KL divergence (DKL) across a normalized across n_gaussians, seq_l and batch

    -DKL = 1/2*sum_1_J(1 + log((sigma)^2) - mu^2 - sigma^2)

    see https://arxiv.org/pdf/1312.6114.pdf - Appendix B

        Parameters:
            mu (batch, seq_l, n_gaussians, latent_dim): means
            sigma (batch, seq_l, n_gaussians, latent_dim): predicted standard deviations
    """
    kl_div_loss = 1/2*torch.sum(torch.ones_like(sigma) + torch.log(torch.square(sigma)) - torch.square(mu) - torch.square(sigma), -1)
    kl_div_loss = kl_div_loss.mean()
    return -kl_div_loss


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
    return sample


if __name__ == "__main__":
    softmax = torch.nn.Softmax(dim=-1)

    mu = torch.rand((2, 10))
    sigma = torch.rand((2, 10))
    pi = softmax(torch.rand(2))

    sample = sample_gmm(pi, mu, sigma)
    print("sample: ", sample)
