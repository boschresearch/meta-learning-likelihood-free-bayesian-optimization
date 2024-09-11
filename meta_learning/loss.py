# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch


def empirical_cdf(
    data: torch.Tensor,
    x
):
    return torch.sum(data < x) / data.size(0)


def _kolmogorov_smirnov_distance(
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Return the kolmogorov distance for each dimension.

    Note that the returned distance are NOT in the same order as embedding matrix.
    Thus, this is useful for means/max, but not for visual inspection.
    """
    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    emb_num, _ = sorted_embeddings.shape[-2:]

    empirical_cdf = torch.linspace(
        start=1 / emb_num,
        end=1.0,
        steps=emb_num,
        device=embedding_matrix.device,
        dtype=embedding_matrix.dtype,
    ).unsqueeze(-1)
    normal_dict_cdf = 0.5 * (1 + torch.erf(sorted_embeddings * 0.70710678118))

    return normal_dict_cdf - empirical_cdf


def mean_squared_kolmogorov_smirnov_distance(
    embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Return the mean-squared Kolmogorov-Smirnov distance over dimensions.

    The idea is to compare the empirical one-dimensional distribution of the
    embeddings across task to the marginal CDFs of a desired prior. The prior
    is assumed to be Gaussian, but test works in general.
    """
    d = _kolmogorov_smirnov_distance(embedding_matrix)

    return torch.mean(d ** 2)


def _empirical_covariance(X: torch.Tensor) -> torch.Tensor:
    X_zeromean = X - torch.mean(X, dim=-2)
    return torch.mm(X_zeromean.t(), X_zeromean) / X.size(-2)


def mean_squared_covariance(embedding_matrix):
    """Compute the mean squared covariance across the rows of embedding_matrix"""
    sigma = _empirical_covariance(embedding_matrix)
    eye = torch.eye(sigma.size(-2), dtype=sigma.dtype, device=sigma.device)

    return torch.nn.functional.mse_loss(sigma, eye)


class MetaLoss:

    def __init__(self, batch_size, latent_dim) -> None:
        self.kolmogorov_weight, self.covariance_weight = self._estimate_embedding_loss_weights(batch_size, latent_dim)        

    def _estimate_embedding_loss_weights(self, batch_size, latent_dim):
        """
        Computes the mean estimated loss for truely Gaussian embeddings
        to scale the embedding loss
        """
        ks_losses = []  # komologorov smirnov loss
        cv_losses = []  # covariance loss

        for _ in range(64):
            random_embeddings = torch.randn([batch_size, latent_dim])
            ks_loss = mean_squared_kolmogorov_smirnov_distance(
                random_embeddings
            ).numpy()
            cv_loss = mean_squared_covariance(random_embeddings).numpy()
            ks_losses.append(ks_loss)
            cv_losses.append(cv_loss)

        kolmogorov_weight = 0.5 / np.mean(ks_losses)
        covariance_weight = 0.5 / np.mean(cv_losses)

        return kolmogorov_weight, covariance_weight

    def _embedding_loss(self, embedding_matrix):
        """
        Loss on the embedding used during training.
        """
        return self.kolmogorov_weight * mean_squared_kolmogorov_smirnov_distance(
            embedding_matrix
        ) + self.covariance_weight * mean_squared_covariance(embedding_matrix)

    def __call__(
        self,
        output,
        targets,
        weights,
        embedding,
        embedding_loss_weight=0.1,
        prediction_loss_weight=1.0
    ):
        """
        Training loss for BaNNER consisting of a loss on the embedding,
        the features and the predictions
        """
        _, predictions = output

        embedding_loss = self._embedding_loss(embedding)
        prediction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            weight=weights
        )

        return (
            embedding_loss_weight * embedding_loss
            + prediction_loss_weight * prediction_loss
        )
