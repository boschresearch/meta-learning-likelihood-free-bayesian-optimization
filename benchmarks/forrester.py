# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math
import copy
from dataclasses import dataclass
from typing import Optional, Dict, Union, Any, List

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification


@dataclass(frozen=True)
class Task:
    uid: Union[str, int]
    """Unique task identifier"""
    descriptors: Dict[str, Any]
    """Parameters for funcitons"""


def forrester_function(x: float, a: float, b: float, c: float) -> float:
    """The one-dimensional Forrester function.

    The function is multimodal, with one global minimum, one local minimum,
    and a zero-gradient inflection point. The function is given by

    .. math::
    f(x) = a f_{high}(x) + b (x - 0.5) - c

    where the high-fidelity Forrester function is given by

    .. math::
    f_{high}(x) = (6x - 2)^2 \sin(12x - 4)

    Parameters:
    -----------
    x: Value of the parameter.
    a, b, c: The parameters for the Forrester function.

    Returns:
    --------
        Observed value at the query point.
    """
    y_high = math.pow(6 * x - 2, 2) * math.sin(12 * x - 4)
    y = a * y_high + b * (x - 0.5) - c
    return y


class Forrester:
    def __init__(
        self, n_data_per_task=[4] * 128, seed: Optional[int] = None
    ):
        self.seed = seed
        self._n_data_per_task = n_data_per_task

        descriptors = ps.ParameterSpace()
        descriptors.add(ps.ContinuousParameter(name="a", bounds=[0.2, 3]))
        descriptors.add(ps.ContinuousParameter(name="b", bounds=[-5, 15]))
        descriptors.add(ps.ContinuousParameter(name="c", bounds=[-5, 5]))

        self._search_space = ps.ParameterSpace()
        self._search_space.add(ps.ContinuousParameter(name="x", bounds=[0, 1]))

        self._target_task, self._meta_tasks = self.create_tasks(
            descriptors, len(n_data_per_task), seed
        )

    @property
    def target_task(self) -> Task:
        return self._target_task

    @property
    def meta_tasks(self) -> Dict[int, Task]:
        return self._meta_tasks

    @property
    def n_data_per_task(self) -> List:
        return self._n_data_per_task

    @property
    def search_space(self) -> ps.ParameterSpace:
        return self._search_space

    @staticmethod
    def create_random_task(
        uid,
        descriptors: ps.ParameterSpace,
        seed: Optional[int] = None,
    ):
        """Return Task instance with randomly initialized parameters."""
        prng = np.random.default_rng(seed)
        return Task(uid, descriptors.sample(rng=prng))

    @staticmethod
    def create_tasks(
        descriptors,
        num_meta_tasks,
        seed: Optional[int] = None,
    ):
        prng = np.random.default_rng(seed)
        target_task = Forrester.create_random_task(0, descriptors, prng)
        meta_tasks = {
            uid: Forrester.create_random_task(uid, descriptors, prng)
            for uid in range(1, num_meta_tasks + 1)
        }
        return target_task, meta_tasks

    def __call__(
        self, 
        eval_spec: EvaluationSpecification,
        task_uid: Optional[int] = None,
    ) -> Evaluation:
        task = self.target_task if task_uid is None else self.meta_tasks[task_uid]

        config = eval_spec.configuration
        objective_values = forrester_function(**config, **task.descriptors)

        return eval_spec.create_evaluation(
            objectives={"loss": objective_values},
            user_info={"task_uid": task_uid}
        )

    def get_meta_data(
        self, seed: Optional[int] = None
    ):
        seed = copy.copy(self.seed) if seed is None else seed
        prng = np.random.default_rng(seed)


        meta_data: Dict[Union[str, int], List[Evaluation]] = dict()

        for (uid, task), n_data in zip(self.meta_tasks.items(), self.n_data_per_task):
            meta_data[uid] = []
            for _ in range(n_data):
                config = self.search_space.sample(rng=prng)
                eval_spec = EvaluationSpecification(configuration=config)
                evaluation = self.__call__(eval_spec, task_uid=uid)
                meta_data[uid].append(evaluation)

        return meta_data
