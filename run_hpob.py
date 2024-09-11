# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.hpob import HPOBBench
from malibo import MALIBO

classifier_config = {
    "num_layers": 5,
    "num_features": 50,
    "num_hidden_units": 64,
    "device": "cpu",
    "dtype": torch.float64,
}
train_config = {
    "num_epochs": 2048,
    "batch_size": 256,
}


def run_optimization_loop(
    benchmark,
    test_seed,
    optimizer,
    max_evaluations: int,
):
    X = np.asarray(benchmark.benchmark_data["X"])
    y = np.asarray(benchmark.benchmark_data["y"])
    y = benchmark.normalize(-y)

    data_size = len(X)
    # indices of pending evaluations
    pending_evaluations = list(range(data_size))
    current_evaluations = []

    init_ids = benchmark.bo_initializations[test_seed]
    for i in range(len(init_ids)):
        idx = init_ids[i]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)

    # NOTE: change max to min
    min_regret_history = [np.min(y[current_evaluations])]
    opt_time = []
    for i in range(max_evaluations):
        # take the acquistion values from the pending evaluations
        start_time = time.time()
        idx = optimizer.observe_and_suggest(
            X[current_evaluations], y[current_evaluations], X[pending_evaluations]
        )
        end_time = time.time()
        opt_time.append(end_time - start_time)
        idx = pending_evaluations[idx]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)
        min_regret_history.append(np.min(y[current_evaluations]))

        if min(y) in min_regret_history:
            break

    # negate to recover accuracy
    min_regret_history += [min(y).item()] * (max_evaluations - i - 1)

    return min_regret_history, opt_time


if __name__ == "__main__":
    with open("benchmarks/HPO-B/hpob-data/meta-test-tasks-per-space.json", "r") as f:
        search_spaces = json.load(f)

    parser = argparse.ArgumentParser(description="Run HPOB benchmarks.")
    # SEARCH_SPACE_ID="4796 5527 5636 5859 5860 5891 5906 5965 5970 5971 6766 6767 6794 7607 7609 5889"
    parser.add_argument(
        "--search_space_id", choices=search_spaces.keys(), nargs="+", required=True
    )
    parser.add_argument("--dataset_id", nargs="+")
    # TEST_SEED="test0 test1 test2 test3 test4"
    parser.add_argument("--test_seed", type=str, required=True)
    parser.add_argument("--continuous", action=argparse.BooleanOptionalAction)
    parser.add_argument("--evaluations", type=int, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Valid options are: test0, test1, test2, test3, test4."
    test_seed = args.test_seed

    # experiment_name = args.name if args.name else args.dataset
    root_dir = Path(args.output) if args.output else Path("./results/hpob/")
    is_continuous = args.continuous

    for search_space_id in args.search_space_id:
        if args.dataset_id is not None:
            datasets = args.dataset_id
        else:
            datasets = search_spaces[search_space_id]

        for dataset_id in datasets:
            benchmark = HPOBBench(
                search_space_id=search_space_id, dataset_id=dataset_id
            )

            optimizer = MALIBO(benchmark.search_space, **classifier_config)
            meta_dir = Path("./checkpoints_hpob") / "MALIBO" / f"{search_space_id}"
            optimizer.save_dir = meta_dir
            if meta_dir.exists():
                optimizer = optimizer.load(meta_dir)
                optimizer.classifier.meta_model.initialize(
                    0,
                    dtype=classifier_config["dtype"],
                    device=classifier_config["device"],
                )
            else:
                meta_data, validation_data = benchmark.get_meta_data()
                # MALIBO does not use the validation set
                # For training MALIBO, each task needs learn a task embedding
                # Testing on validation data without training on it is not possible
                optimizer.meta_fit(meta_data, meta_dir=meta_dir, **train_config)

            # run BO loop
            regret, opt_time = run_optimization_loop(
                benchmark=benchmark,
                test_seed=test_seed,
                optimizer=optimizer,
                max_evaluations=args.evaluations,
            )

            output_dir = root_dir / test_seed / "MALIBO" / search_space_id
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"regret": regret}).to_csv(output_dir / f"{dataset_id}.csv")
            pd.DataFrame({"time": opt_time}).to_csv(
                output_dir / f"{dataset_id}_opt_time.csv"
            )
