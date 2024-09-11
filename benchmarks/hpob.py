# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import sys
sys.path.append('benchmarks/HPO-B')

from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective

from hpob_handler import HPOBHandler


def get_parameterspace(
    benchmark, search_space_id
) -> ps.ParameterSpace:
    space = ps.ParameterSpace()

    dataset = list(benchmark.meta_test_data[search_space_id].keys())[0]
    X = benchmark.meta_test_data[search_space_id][dataset]["X"][0]
    search_space_dims = len(X)

    for n in range(search_space_dims):
        # default parameter type is "uniform_float"
        space._parameters[f"x_{n}"] = {
            "parameter": ps.ContinuousParameter(
                name=f"x_{n}",
                bounds=(0., 1.),
                transformation=None,
            ),
            "condition": ps.Condition(),
        }
    return space


class Tabular:
    def __init__(self, data) -> None:
        self.data = pd.DataFrame(data)

    def predict(self, config):
        idx = np.all(self.data['X'].tolist() == config, axis=1).nonzero()[0].item()
        return self.data['y'][idx][0]

    def get_minimum(self):
        return self.data['y'].min()[0]

    def get_maximum(self):
        return self.data['y'].max()[0]


class HPOBBench:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        input_dir: str = './benchmarks/HPO-B',
        seed: Optional[int] = None,
        fidelity: float = 1.0,
        objectives: Optional[List[Objective]] = None,
        continuous: bool = False,
    ):
        """
        dataset_name: search_space id in the paper, e.g. alogorithm
        """
        self.root_dir = Path(input_dir)
        self.data_dir = self.root_dir / "hpob-data"
        self.surrogates_dir = str(self.root_dir / 'saved-surrogates')
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id
        self.fidelity = fidelity
        self.seed = seed
        self.continuous = continuous

        self._benchmark, self._benchmark_model = self._get_benchmark(continuous)
        self._search_space = get_parameterspace(self._benchmark, search_space_id)
        self.search_space_dim = len(self.search_space)
        self._objectives = (
            [Objective("loss", greater_is_better=False)]
            if objectives is None
            else objectives
        )

    @property
    def objectives(self) -> List[Objective]:
        return self._objectives

    @property
    def search_space(self):
        return self._search_space

    @property
    def output_dimensions(self) -> int:
        return len(self.objectives)

    def get_minimum(self):
        if self.continuous:
            return self.surrogates_stats[self.surrogate_name]["y_min"]
        else:
            return self._benchmark_model.get_minimum()

    def get_maximum(self):
        if self.continuous:
            return self.surrogates_stats[self.surrogate_name]["y_max"]
        else:
            return self._benchmark_model.get_maximum()

    @property
    def bo_initializations(self):
        return self._benchmark.bo_initializations[self.search_space_id][self.dataset_id]

    @property
    def benchmark_data(self):
        return self._benchmark.meta_test_data[self.search_space_id][self.dataset_id]

    def normalize(self, *args, **kwargs):
        return self._benchmark.normalize(*args, **kwargs)

    def get_meta_data(self):
        handler = HPOBHandler(
            root_dir=self.data_dir,
            mode='v3',
            surrogates_dir=self.surrogates_dir
        )
        train_dataset_ids = list(handler.meta_train_data[self.search_space_id].keys())

        meta_data: Dict[Union[str, int, List[Evaluation]]] = dict()
        for uid, dataset_id in enumerate(train_dataset_ids, start=1):
            meta_data[uid] = []
            # get meta-training data
            xs = np.array(handler.meta_train_data[self.search_space_id][dataset_id]['X'])
            ys = np.array(handler.meta_train_data[self.search_space_id][dataset_id]['y'])

            # turn into minimization problem
            for x, y in zip(xs, -ys):
                config = {}
                for n, x_n in enumerate(x):
                    config[f'x_{n}'] = x_n
                eval_spec = EvaluationSpecification(configuration=config)
                evaluation = eval_spec.create_evaluation(
                    objectives={"loss": y.item()},
                    user_info={
                        "task_uid": dataset_id,
                        "cost": None
                    },
                )
                meta_data[uid].append(evaluation)

        validation_dataset_ids = list(handler.meta_validation_data[self.search_space_id].keys())
        validation_data: Dict[Union[str, int, List[Evaluation]]] = dict()
        for uid, dataset_id in enumerate(validation_dataset_ids, start=1):
            validation_data[uid] = []
            # get meta-validation data
            xs = np.array(handler.meta_validation_data[self.search_space_id][dataset_id]['X'])
            ys = np.array(handler.meta_validation_data[self.search_space_id][dataset_id]['y'])

            for x, y in zip(xs, -ys):
                config = {}
                for n, x_n in enumerate(x):
                    config[f'x_{n}'] = x_n
                eval_spec = EvaluationSpecification(configuration=config)
                evaluation = eval_spec.create_evaluation(
                    objectives={"loss": y.item()},
                    user_info={
                        "task_uid": dataset_id,
                        "cost": None
                    },
                )
                validation_data[uid].append(evaluation)

        return meta_data, validation_data
    
    def _get_benchmark(self, continuous):
        # NOTE: load test data only
        handler = HPOBHandler(
            root_dir=self.data_dir,
            mode='v3-test',
            surrogates_dir=self.surrogates_dir
        )
        if continuous:
            raise NotImplementedError("Continuous search space not supported yet.")

        else:
            # normalized X and y
            xs = np.array(handler.meta_test_data[self.search_space_id][self.dataset_id]["X"])
            ys = np.array(handler.meta_test_data[self.search_space_id][self.dataset_id]["y"])
            # NOTE: negate to be minimized
            ys = handler.normalize(-ys)

            data = {'X': xs.tolist(), 'y': ys.tolist()}
            tabular = Tabular(data)

            return handler, tabular
