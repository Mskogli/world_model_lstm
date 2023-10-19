import torch

from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent


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

    normal_loglik = (
        -1
        / 2
        * (
            torch.einsum("bsdc, bsdc ->bsd", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
    )

    loglik = (
        1 / logpi.size(1) * torch.logsumexp(logpi + normal_loglik, dim=2)
    )  # GMM LL normalized by sequence length

    loglik = loglik.mean()

    return -loglik


def KL_divergence_loss(
    logpi: torch.tensor, mu: torch.tensor, sigma: torch.tensor
) -> float:
    """
    Calculates the KL-divergence between p and q where p is an arbitrary gaussian distribution and q is the a gaussian with zero mean and unit variance
    KL = 1/2*sum_1_J(1 + log((sigma)^2) - mu^2 + sigma^2)

    see https://arxiv.org/pdf/1312.6114.pdf - Appendix B
    """

    pass


def sample_gmm(pi: torch.tensor, mu: torch.tensor, sigma: torch.tensor) -> torch.tensor:
    """
    Sample from a gaussian mixture model

        Parameters:
            pi (n_gaussians)
            mu (n_gaussians, gaussian_dim)
            sigma (n_gaussians, gaussian_dim)
    """

    categorical = Categorical(probs=pi)
    # gaussians = [Normal(loc=mu[i], scale=sigma[i]) for i in range(pi.shape[0])]

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
