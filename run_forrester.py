# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from benchmarks.forrester import Forrester
from malibo import MALIBO
from plotting import plot_features, plot_update

classifier_config = {
    "num_layers": 5,
    "num_features": 50,
    "num_hidden_units": 64,
    "device": "cpu"
}
train_config = {
    "num_epochs": 2048,
    "batch_size": 256,
}


if __name__ == "__main__":
    benchmark = Forrester(n_data_per_task=[32] * 256, seed=44)
    meta_data = benchmark.get_meta_data()

    optimizer = MALIBO(benchmark.search_space, **classifier_config)
    optimizer.meta_fit(meta_data, **train_config)
    plot_features(benchmark, optimizer)

    num_steps = 16
    for i in range(num_steps):
        eval_spec = optimizer.generate_evaluation_specification()
        evaluation = benchmark(eval_spec)
        optimizer.report(evaluation)
        plot_update(
            benchmark,
            optimizer,
            optimizer.seed,
            title=f"Iteration {i+1}"
        )
