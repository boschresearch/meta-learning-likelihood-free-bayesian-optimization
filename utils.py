# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
from typing import Optional, Generator
from contextlib import contextmanager

import torch
import numpy as np


def configure_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = 0
    # prevent adding duplicates to logger
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s | %(message)s')
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


@contextmanager
def manual_seed(seed: Optional[int] = None) -> Generator[None, None, None]:
    """
    Contextmanager for manual setting the torch.random seed.

    Args:
        seed: The seed to set the random number generator to.

    Returns:
        Generator
    """
    old_state = torch.random.get_rng_state()
    try:
        if seed is not None:
            torch.random.manual_seed(seed)
        yield
    finally:
        if seed is not None:
            torch.random.set_rng_state(old_state)


def metadata_to_training(meta_data):
    """Convert data to standard (X, y) input-output format.

    Parameters
    ----------
    metadata : dict
        Dictionary of task IDs that contain the data.
        See `metafidelity.benchmarks.base.BaseBenchmark.get_meta_data`
        for a detailed description of the output.

    Returns
    -------
    X : list of ndarrays
        Training data, with one array per task in the list.
    y : list of ndarrays
        Training data, with one array per task in the list.
    """
    # Transform metadata into a list of arrays,
    X = []
    y = []
    w = []
    task_ids = []
    for task_id, data in meta_data.items():
        num_obs = len(data["X"])
        X.extend(data["X"])
        y.extend(data["Y"])
        w.extend(data["w"])
        task_ids.extend(np.ones(num_obs, dtype=int) * task_id)

    return np.array(X), np.array(task_ids), np.array(y), np.array(w)
