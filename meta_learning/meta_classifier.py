# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import importlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator

from inference.laplace_approximation import BLRLayer
from meta_learning.meta_model import TaskAgnosticModel
from meta_learning.trainer import train
from utils import configure_logger, metadata_to_training


class MetaBLOR(BaseEstimator):
    """
    Meta-learning Bayesian Logistic Regression

    This classifier contains two components:
        1. A task-agnostic model (g_w), which meta-learns on the meta-data
        and fixed during inference. It provides the features representation
        shared across task and mean predicition for warm-starting optimization. 
        2. A task-specific layer parameterized by z, which is a Bayesian
        logistic regression using the meta-learned features shared across
        task. It performs adaptation on the target task.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: Optional[str]=None,
        dtype=torch.float64,
        **model_config
    ):
        super().__init__()
        self.input_dim = input_dim,
        self.output_dim = output_dim
        self.model_config = model_config
        self._init_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            **model_config
        }

        # assign device automatically
        self.dtype = dtype
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tkwargs = {"dtype": self.dtype, "device": self.device}

        # Task-agnostic meta-learning
        self.meta_model = TaskAgnosticModel(
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config
        ).to(**self.tkwargs)
        self.blr_layer = self._blr_layer_init()

        self.logger = configure_logger(__name__)

    def _blr_layer_init(self):
        """Initialize the weights of Bayesian logistic regression"""

        blr_layer = BLRLayer(
            self.model_config["num_features"],
            output_dim=self.output_dim,
            **self.tkwargs
        )
        return blr_layer

    def get_features_and_mean_logits(self, X):
        """Return the learned features and mean logits given input X"""

        embedding = torch.tensor(0, device=self.device, dtype=torch.long)
        with torch.no_grad():
            features, mean_logits = self.meta_model(X, embedding)
        return features, mean_logits

    def meta_fit(
        self,
        meta_data: Dict,
        num_epochs: int,
        batch_size: int,
        **train_config
    ):
        """Fit model to training data.

        Args:
            meta_data: Dictionary with meta-data.
        """
        # transform dictionary containing meta-data to suitable format
        # X, y: observations in related task, y is the class label
        # task_ids: task identifier indicating which related task the data from
        # w: weights computed by EI utility function
        X, task_ids, y, w = metadata_to_training(meta_data)

        train(
            model=self.meta_model,
            data=(X, task_ids, y, w),
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_data=None,
            device=self.device,
            **train_config
        )

    def fit(self, X, y, sample_weight, **kwargs):
        """Fit the task-specific Bayesian logistic regression on target task data"""

        self.meta_model.eval().to(**self.tkwargs)
        X = torch.tensor(X, **self.tkwargs)
        y = torch.tensor(y, **self.tkwargs).reshape(-1, self.output_dim)
        sample_weight = torch.tensor(sample_weight, **self.tkwargs).reshape(-1, 1)

        self._fit(X, y, sample_weight, **kwargs)

    def _fit(self, X, y, sample_weight):
        """Fit the task-specific Bayesian logistic regression on target task data"""

        train_tensor = [X, y, sample_weight]
        train_dataset = torch.utils.data.TensorDataset(*train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))

        # For every optimization iteration, we train a new Bayesian logisitic regression
        # classifier to adapt on the target task, therefore initialize the weights in the layer.
        self.blr_layer = self._blr_layer_init()

        # Use LBFGS to optimize the weights 
        optimizer = torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, self.blr_layer.parameters()),
            line_search_fn="strong_wolfe"
        )

        def compute_loss(dataloader, meta_model, blr_layer):
            mle_loss = torch.tensor([0.], **self.tkwargs)
            for X, y, w in dataloader:
                embedding = torch.tensor(0, device=self.device, dtype=torch.long)
                with torch.no_grad():
                    features, mean_logits = meta_model(X, embedding)
                logits = blr_layer(features, mean_logits, y, w)
                mle_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, weight=w, reduction='sum'
                )
            prior_loss = torch.tensor([0.], **self.tkwargs)
            for param in blr_layer.parameters():
                if param.requires_grad:
                    prior_loss += 0.5 * torch.matmul(param, param.t()).squeeze()
            return mle_loss + prior_loss

        def closure():
            optimizer.zero_grad()
            self.blr_layer.reset_covariance_matrix()
            loss = compute_loss(
                train_dataloader,
                self.meta_model,
                self.blr_layer,
            )
            loss.backward()
            return loss

        self.blr_layer.train()
        optimizer.step(closure)
        self.blr_layer.eval()

    def predict_proba(self, X, *args, **kwargs):
        X = np.array(X, dtype=np.float64)
        pos_prob = self.predict(X, *args, **kwargs)
        return np.column_stack((1 - pos_prob, pos_prob))

    def predict(self, X, *args, **kwargs):
        X = torch.tensor(X, **self.tkwargs)
        with torch.no_grad():
            mean = self._predict(X, *args, **kwargs)

        mean = mean.detach().cpu().numpy()
        return mean

    def _predict(self, X, sampling='thompson_sampling'):
        self.meta_model.eval().to(**self.tkwargs)
        self.blr_layer.eval()

        embedding = torch.tensor(0, device=self.device, dtype=torch.long)
        features, mean_logits = self.meta_model(X, embedding)

        if sampling == 'max':
            logits = self.blr_layer(features, mean_logits)
        elif sampling == 'thompson_sampling':
            logits = self.blr_layer.sample(features, mean_logits)
        else:
            raise ValueError(f"{sampling} is not supported.")

        pred_probs = torch.sigmoid(logits)
        return pred_probs

    def save(self, save_dir):
        """Save the meta-model for later use."""
        # saving pytorch model
        torch.save(self.meta_model.state_dict(), save_dir / "meta_model.torch")

        config = {
            "init_kwargs": self._init_kwargs,
            "model_type": type(self.meta_model).__module__ + "." + type(self.meta_model).__qualname__,
        }
        with open(save_dir / "meta_model_config.json", "w") as f:
            json.dump(config, f)
        self.logger.info(f"Saved meta-learned model to {save_dir}.")

    def load(self, load_dir, strict=False, **kwargs):
        """Load meta model from disk to fit to new task"""
        if not load_dir.exists():
            raise FileNotFoundError(
                f"The load directory does not exist: "
                f"{load_dir}"
            )
        load_dir = Path(load_dir)

        with open(load_dir / "meta_model_config.json", "r") as f:
            config = json.load(f)
        config["init_kwargs"].update(kwargs)

        model_type = config["model_type"]
        module_str, class_str = model_type.rsplit(".", 1)
        module = importlib.import_module(module_str)
        model_class = getattr(module, class_str)
        self.meta_model = model_class(**config["init_kwargs"])

        self.meta_model.load_state_dict(
            torch.load(load_dir / "meta_model.torch", weights_only=False),
            strict=strict
        )
        self.meta_model.to(**self.tkwargs)
        self.meta_model.eval()
        self.logger.info(f"Loaded meta-learned model from {load_dir}.")
        return self
