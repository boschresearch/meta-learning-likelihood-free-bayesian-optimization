# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import parameterspace as ps
from blackboxopt.evaluation import Evaluation, EvaluationSpecification
from sklearn.ensemble import GradientBoostingClassifier

from meta_learning.meta_classifier import MetaBLOR
from utils import manual_seed


class RandomSearchSampler:
    def __init__(self, search_space: ps.ParameterSpace, seed: Optional[int] = None):
        """Random search for configurations"""
        self.search_space = search_space.copy()
        self.seed = seed
        self.search_space.seed(seed)

    def __call__(self) -> EvaluationSpecification:
        return EvaluationSpecification(configuration=self.search_space.sample())


class MALIBO:
    """Meta-learning for likelihood-free Bayesian optimization
    Args:
        search_space: Search space of the problem
        seed: Random seed
        gamma: Hyperparameter in LFBO, which is the percentile of configurations
            that are considered as good. It controls the exploration-exploitation,
            default is 0.33.
        num_samples_acquisition_function: Number of random samples for optimizing the
            acquisition function.
    """
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        seed: Optional[int] = None,
        gamma: float = 1/3,
        num_samples_acquisition_function: int = 5120,
        **classifier_kwargs,
    ):

        self.search_space = search_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.num_samples_acquisition_function = num_samples_acquisition_function
        self.random_sampler = RandomSearchSampler(search_space, seed=seed)

        # meta-learning classifier
        self.classifier = MetaBLOR(
            input_dim=len(search_space),
            output_dim=1,
            **classifier_kwargs
        )
        # gradient boosting classifier
        self.classifier_gb = None

        # saved previous observations
        self.X = []
        self.losses = []

    @staticmethod
    def _reweight_samples(X, y, gamma, weight_type='ei'):
        """Assigning class labels to observations with the threshold tau computed via gamma.
        Weights are computed by the utility function max(tau - y, 0) from expected improvement (EI).   
        Args:
            X: Observations inputs
            y: Observations values
            gamma: Hyperparameter in LFBO, which is the percentile of configurations
                that are considered as good. It controls the exploration-exploitation,
                default is 0.33.
            weight_type: Determine which utility function to use. EI uses max(tau - y, 0) while PI
                uses 1(tau - y > 0).
        Returns:
            X: Observations inputs
            z: Class labels
            w: Weights calculated by utility function
        """
        tau = np.quantile(np.unique(y), q=gamma)
        z = np.less(y, tau)

        if weight_type == 'ei' and len(X) > 1:
            z_idx = z.squeeze()

            x1, z1 = X[z_idx], z[z_idx]
            x0, z0 = X, np.zeros_like(z)

            w1 = (tau - y)[z_idx]
            w1 = w1 / np.mean(w1) if len(w1) else w1
            w0 = 1 - z0

            x = np.concatenate([x1, x0], axis=0)
            z = np.concatenate([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / np.mean(w)

        elif weight_type == 'pi' or len(X) == 1:
            x = X
            w = np.ones_like(z)

        return x, z.astype(float), w.squeeze()

    def meta_fit(
        self,
        meta_data: Optional[Dict[str, List[Evaluation]]] = None,
        num_epochs: int = 2048,
        batch_size: int = 256,
        meta_dir: Optional[str] = None,
        override: bool = True,
        **train_config
    ):
        """Meta-learning on meta-data, corresponds to the meta-learning part in Algorithm 1."""
        converted_meta_data = dict()
        for task_uid, evaluations in meta_data.items():
            X = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations])
            Y = np.array([e.objectives["loss"] for e in evaluations]).reshape(-1, self.classifier.output_dim)
            X, Y, w = self._reweight_samples(X, Y, self.gamma)
            converted_meta_data[task_uid] = {"X": X, "Y": Y, "w": w.reshape(-1, 1)}

        self.classifier.meta_fit(
            converted_meta_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            **train_config
        )
        if meta_dir is not None:
            self.save(meta_dir, override=override)

    def _update_model(self, weight_type='ei'):
        """Train optimizer on new observations from the target task.
        """
        # first transform the observation values (self.losses) to class labels z,
        # with weight w computed through EI utility function 
        x, z, w = self._reweight_samples(
            self.X,
            self.losses,
            self.gamma,
            weight_type
        )

        # Since early stopping for gradient boosting required at least 2 obs in each class,
        # we only train the gradient boosting classifier after obtaining enough observations
        if sum(z == 0.) >= 2 and sum(z == 1.) >= 2:
            # train a GB with early stopping to estimate n_estimators
            classifier_gb = GradientBoostingClassifier(
                init=self.classifier,
                # subsample=0.8,
                validation_fraction=0.3,
                n_iter_no_change=5,
            )
            classifier_gb.fit(x, z, sample_weight=w)
            # train on full dataset with estimated n_estimators
            self.classifier_gb = GradientBoostingClassifier(
                init=self.classifier,
                n_estimators=classifier_gb.n_estimators_
            )
            # train classifier weighted by utility
            self.classifier_gb.fit(x, z, sample_weight=w)
        else:
            self.classifier.fit(x, z, sample_weight=w)

    def generate_evaluation_specification(self, sampling='thompson_sampling', **kwargs):
        """
        Propose new configuration x to evaluate.

        1. First proposal is proposed according to the maximum of the meta-learned
        acquisition function.
        2. After the first evaluation, the proposed candidate is selected based on
        the Thompson sample of the acquisition function. 
        """

        if len(self.X) == 0:
            configuration = self._sample_new_config(sampling="max", **kwargs)
        else:
            self._update_model()
            configuration = self._sample_new_config(sampling=sampling, **kwargs)
        eval_spec = EvaluationSpecification(configuration=configuration)

        # if new proposed configuration is duplicate, we used a random sample as a
        # substitution
        if self._is_duplicate(self.search_space.to_numerical(eval_spec.configuration)):
            eval_spec = self.random_sampler()

        return eval_spec

    def observe_and_suggest(
        self,
        X_obs,
        y_obs,
        X_pen=None,
        sampling='thompson_sampling',
        seed=None,
        **kwargs
    ):
        self.X = X_obs
        self.losses = y_obs.ravel()
        rng = np.random.default_rng(seed)
        if seed is None:
            seed = rng.integers(32767, size=1)
            self.seed = seed

        # X_pen not None for HPOB discrete
        if X_pen is not None:
            if len(np.unique(self.losses)) <= 1:
                scores = self.predict(X_pen, sampling='max', seed=seed, **kwargs)
                candidate = np.argmax(scores)
                return candidate
            else:
                self._update_model()
                scores = self.predict(X_pen, sampling=sampling, seed=seed, **kwargs)
                candidate = np.argmax(scores)
                return candidate
        else:
            if len(np.unique(self.losses)) <= 1:
                configuration = self._sample_new_config(sampling="max", **kwargs)
                new_x = self.search_space.to_numerical(configuration)
                return new_x
            else:
                self._update_model()
                configuration = self._sample_new_config(sampling=sampling, **kwargs)
                # if new proposed configuration is duplicate, the we random sample one
                if self._is_duplicate(self.search_space.to_numerical(configuration)):
                    configuration = self.search_space.sample()
                new_x = self.search_space.to_numerical(configuration)
                return new_x

    def predict(self, X, sampling="thompson_sampling", seed=None):
        """Predict the utility of given input X (acquisition function value for given points)"""
        # In order to visualize and reproduce the Thompson sampling result later,
        # we use a random seed to generate samples and save the random seed.
        with manual_seed(seed):
            if self.classifier_gb is None:
                class_probability = self.classifier.predict(X, sampling=sampling)
            else:
                class_probability = self.classifier_gb.predict_proba(X)[:, 1]
        return class_probability

    def _sample_new_config(
        self,
        sampling,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Optimizing the acquisition function.

        As pointed out by the BORE paper, random search for optimizing the
        acquisition function generated by tree-based methods is better than
        using evolutionary algorithm. Therefore we use random search for
        optimizing the acquisition function. 
        """
        rng = np.random.default_rng(seed)
        samples = rng.random([
            self.num_samples_acquisition_function, len(self.search_space)
        ])
        if seed is None:
            # generate random seed to produce Thompson samples
            seed = rng.integers(32767, size=1)
            self.seed = seed

        af_samples = self.predict(samples, sampling=sampling, seed=seed, **kwargs)
        best_vector = samples[np.argmax(af_samples)]

        return self.search_space.from_numerical(best_vector)

    def _is_duplicate(self, new_X):
        return any(np.array_equal(x, new_X) for x in self.X)

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """Report observations to the optimizer"""
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        for evaluation in _evals:
            self._report(evaluation)

    def _report(self, evaluation: Evaluation) -> None:
        new_X = np.atleast_2d(self.search_space.to_numerical(evaluation.configuration))
        new_loss = evaluation.objectives["loss"]

        if len(self.X) > 0:
            self.X = np.vstack([self.X, new_X])
            self.losses = np.concatenate([self.losses, [new_loss]])
        else:
            self.X = new_X
            self.losses = np.array([new_loss])

    def save(self, save_dir, override=False):
        save_dir = Path(save_dir)
        if save_dir.exists():
            if not save_dir.is_dir():
                raise NotADirectoryError(
                    f'The directory to which to save the model is an existing file: '
                    f'"{save_dir}"'
                )
            if not override:
                raise FileExistsError(
                    f'The directory to which to save the model, already exists: '
                    f'"{save_dir}"'
                )
            else:
                shutil.rmtree(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        self.classifier.save(save_dir)

    def load(self, load_dir, **kwargs):
        load_dir = Path(load_dir)
        self.classifier = self.classifier.load(load_dir, **kwargs)
        return self
