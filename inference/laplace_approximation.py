# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math
from typing import List

import torch


class BLRLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dtype=torch.float64,
        device=torch.device("cpu"),
        **blr_output_kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.blr_output_kwargs = blr_output_kwargs

        self._cov_layer = LaplaceCovariance(device=device, dtype=dtype)
        self._cov_layer.initialize(input_dim)
        self._output_layer = torch.nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
            **self.blr_output_kwargs
        ).to(device=device, dtype=dtype)
        self._output_layer.weight.data.zero_()

    def reset_covariance_matrix(self):
        """Resets covariance matrix of the GP layer.
        This function is useful for reseting the model's covariance matrix
        at the beginning of a new epoch.
        """
        self._cov_layer.initialize_precision_matrix()

    def forward(self, feature, mean_logit, y=None, sample_weights=None):
        # Computes posterior center (i.e., MAP estimate) and variance.
        res_logit = self._output_layer(feature)
        covmat = self._cov_layer(
            feature, res_logit.clone().detach(), mean_logit, y, sample_weights
        )
        if self.training:
            return res_logit + mean_logit
        else:
            logits_adjusted = probit_approximation(res_logit + mean_logit, covmat)
            return logits_adjusted

    def sample(self, feature, mean_logit, num_samples: List = [1]):
        # Thompson sampling
        w_map = self._output_layer.weight.data
        cov_matrix = torch.linalg.inv(self._cov_layer.precision_matrix)
        w_mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            w_map.flatten(), cov_matrix
        )
        w_sample = w_mvn.sample(num_samples)
        logits_sample = torch.mm(feature, w_sample.t()) + mean_logit
        return logits_sample


class LaplaceCovariance(torch.nn.Module):
    """
    Compute the covariance of the task embedding posterior using Laplace method.
    """
    def __init__(
        self,
        dtype=None,
        device="cpu",
    ):
        self.dtype = dtype
        self.device = device
        super().__init__()

    def initialize(self, feature_dim):
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (
            torch.eye(feature_dim, dtype=self.dtype, device=self.device)
        )
        self.precision_matrix = torch.nn.parameter.Parameter(
            torch.eye(feature_dim, dtype=self.dtype, device=self.device),
            requires_grad=False
        )
        self.register_parameter('precision_matrix', self.precision_matrix)

    def update_precision_matrix(
        self,
        features,
        res_logits,
        mean_logits,
        y,
        sample_weights,
        precision_matrix
    ):
        """Update the precision matrix of the BLR layer using Eq.7 in the paper"""
        prob = torch.sigmoid(res_logits + mean_logits)
        prob_multiplier = prob * (1. - prob) * (y * sample_weights + 1.)

        features_adjusted = torch.sqrt(prob_multiplier) * features
        precision_matrix_batch = torch.matmul(
            features_adjusted.t(), features_adjusted
        )
        # compute precision matrix using batch update
        precision_matrix_new = precision_matrix + precision_matrix_batch 

        return precision_matrix_new

    def initialize_precision_matrix(self):
        """Initialize the precision matrix."""
        with torch.no_grad():
            self.precision_matrix.data = self.initial_precision_matrix

    def compute_predictive_covariance(self, features):
        """Computes predictive variance."""
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, features.t())
        cov_matrix = torch.matmul(features, cov_feature_product)
        return cov_matrix

    def forward(self, inputs, res_logits=None, mean_logits=None, y=None, sample_weights=None):
        """updates the posterior precision matrix estimate."""
        batch_size = inputs.shape[0]

        if self.training:
            # Define and register the update op for feature precision matrix.
            if sample_weights is None:
                raise ValueError("Sample weights should not be None.")
            with torch.no_grad():
                self.precision_matrix.data = \
                    self.update_precision_matrix(
                        features=inputs,
                        res_logits=res_logits,
                        mean_logits=mean_logits,
                        y=y,
                        sample_weights=sample_weights,
                        precision_matrix=self.precision_matrix
                    )
            # Return null estimate during training.
            return torch.eye(batch_size, dtype=self.dtype, device=self.device)
        else:
            # Return covariance estimate during inference.
            return self.compute_predictive_covariance(features=inputs)


def probit_approximation(logits, covariance_matrix=None, numbda_square=math.pi/8.):
    # Compute standard deviation.
    if covariance_matrix is None:
        variances = 1.
    else:
        variances = torch.linalg.diagonal(covariance_matrix)

    # Compute scaling coefficient for the approximation.
    logits_scale = torch.sqrt(1. + variances * numbda_square)

    if len(logits.shape) > 1:
        logits_scale = logits_scale[..., None]

    return logits / logits_scale
