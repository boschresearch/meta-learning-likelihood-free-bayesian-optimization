# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from blackboxopt.evaluation import EvaluationSpecification

from utils import manual_seed

params = {
    'axes.labelsize': 11,
    'axes.titlesize': 15,
    'legend.fontsize': 13,
    'font.size': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.edgecolor': "black",
    'axes.linewidth': 1.5,
    'font.family': 'serif',
}
plt.rcParams.update(params)


def plot_obs(x, y, ax, gamma=0.33):
    c = np.arange(len(x))
    ax.scatter(
        x[:-1], y[:-1], c=c[:len(x) - 1], cmap='viridis_r', s=30, label="observations"
    )
    ax.scatter(
        x[-1:], y[-1:], marker="*", s=100, color="tab:orange", label="current query"
    )


def plot_metadata(benchmark, meta_data, gamma, ax):
    x_numerical = np.linspace(0, 1, 100).reshape(-1, 1)
    config_dense = np.array([benchmark.search_space.from_numerical(x)['x'] for x in x_numerical])
    eval_specs = [
        EvaluationSpecification(configuration={'x': config}) for config in config_dense
    ]
    e_target = [benchmark(eval_spec) for eval_spec in eval_specs]
    y_target = [e.objectives['loss'] for e in e_target]

    for task_uid, evaluations in meta_data.items():
        e_meta = [benchmark(eval_spec, task_uid=task_uid) for eval_spec in eval_specs]
        y_meta = [e.objectives['loss'] for e in e_meta]

        xs = np.array([e.configuration['x'] for e in evaluations])
        ys = np.array([e.objectives["loss"] for e in evaluations])

        tau = np.quantile(np.unique(ys), q=gamma)
        labels = np.less(ys, tau).squeeze()

        ax.scatter(
            xs[labels],
            ys[labels],
            marker='x',
            s=20,
            color="tab:red",
            label=r'observations $y \leq \tau$' if task_uid==1 else None
        )
        ax.scatter(
            xs[~labels],
            ys[~labels],
            marker='x',
            s=20,
            color="tab:blue",
            label=r'observations $y > \tau$' if task_uid==1 else None
        )
        ax.plot(
            config_dense, y_meta, color='gray', alpha=.3,
            label='related tasks' if task_uid==1 else None
        )

        if task_uid > 20:
            break

    ax.plot(config_dense, y_target, color='k', label="target task")


def plot_features(benchmark, optimizer, seed=[42], figsize=(10, 6), root_dir="results/forrester"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax_func = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)

    x_numerical = np.linspace(0, 1, 100).reshape(-1, 1)
    configs_dense = np.array([benchmark.search_space.from_numerical(x)['x'] for x in x_numerical])

    # plot meta data
    meta_data = benchmark.get_meta_data()
    plot_metadata(benchmark, meta_data, gamma=0.33, ax=ax_func)
    ax_func.set_ylabel(r'$f(x)$')
    ax_func.set_xlim(-0.01, 1.01)
    ax_func.set_title("Objective function and meta-data")
    ax_func.set_xticks([])

    # plot acquisition function
    ax_af = plt.subplot2grid((3, 4), (2, 0), rowspan=1, colspan=2)
    z_pred = optimizer.predict(x_numerical, sampling='max')
    ax_af.plot(configs_dense, z_pred, label="mean prediction", color='tab:blue')
    ax_af.fill_between(
        configs_dense.flatten(),
        z_pred.flatten(),
        np.zeros_like(z_pred.flatten()),
        alpha=0.3,
        color='tab:blue'
    )

    # plot Thompson samples
    ts_seeds = np.random.randint(0, 1000, 4)
    ts_seeds = np.concatenate((seed, ts_seeds))
    for n, s in enumerate(ts_seeds):
        with manual_seed(s):
            ts_predictions = optimizer.predict(x_numerical, sampling='thompson_sampling', seed=s)
        label = "Thompson samples" if n < 1 else None
        ax_af.plot(configs_dense, ts_predictions, label=label, linestyle='--', alpha=0.7)
    ax_af.set_title("Acquisition function")
    ax_af.set_ylabel(r'$p(y \leq \tau \mid x)$')
    ax_af.set_ylim(-0.01, 1.01)
    ax_af.set_xlim(-0.01, 1.01)
    ax_af.set_xlabel(r'$x$')

    # plot features
    features, _ = optimizer.classifier.get_features_and_mean_logits(
        torch.tensor(x_numerical, device=optimizer.classifier.device)
    )
    features = features.detach().cpu().numpy()
    n_fig_per_row = 2
    ind = np.random.choice(features.shape[-1], 6)
    for i in range(3*n_fig_per_row):
        idx = i
        row = int(idx // n_fig_per_row)
        col = int(idx % n_fig_per_row)

        ax_features = plt.subplot2grid((3, 4), (row, col + 2), rowspan=1, sharex=ax_func if row < 2 else None)
        ax_features.set_xlim(-0.01, 1.01)
        ax_features.plot(
            configs_dense,
            features[:, ind[i]].reshape(configs_dense.shape),
        )

        ax_features.set_title(f"feature {idx+1}")
    ax_features.set_xlabel(r'$x$')

    handles_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0, -.13, 1, -.13), frameon=False)
    plt.tight_layout()
    plt.savefig(root_dir / "malibo_features.png")


def plot_update(benchmark, optimizer, seed, title=None, root_dir="results/forrester"):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax_func = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    x_numerical = np.linspace(0, 1, 1000).reshape(-1, 1)
    configs_dense = np.array([[optimizer.search_space.from_numerical(s)['x']] for s in x_numerical])
    eval_specs = [EvaluationSpecification(configuration={'x': config}) for config in configs_dense]
    e_dense = [benchmark(eval_spec) for eval_spec in eval_specs]
    y_dense = [e.objectives['loss'] for e in e_dense]
    
    # plot target function
    ax_func.plot(configs_dense, y_dense, color='k', label='target function')

    # plot observation
    configs = np.array([[optimizer.search_space.from_numerical(s)['x']] for s in optimizer.X])
    plot_obs(configs, optimizer.losses, ax=ax_func)
    ax_func.set_title(title)

    # plot acquisition function
    ax_af = plt.subplot2grid((3, 1), (2, 0), sharex=ax_func)
    predictions = optimizer.classifier.predict(x_numerical, sampling='max')
    ax_af.plot(configs_dense, predictions, label="mean prediction", color='tab:blue')
    ax_af.fill_between(
        configs_dense.flatten(),
        predictions.flatten(),
        np.zeros_like(predictions.flatten()),
        alpha=0.3,
        color='tab:blue'
    )

    # plot Thompson samples
    if len(optimizer.X) > 1:
        with manual_seed(seed[0]):
            ts_predictions = optimizer.classifier.predict(x_numerical, sampling='thompson_sampling')
        ax_af.plot(configs_dense, ts_predictions, label="Thompson samples", linestyle='--')

    # plot gradient boosting results
    if optimizer.classifier_gb is not None:
        predictions_gb = optimizer.predict(x_numerical, sampling='thompson_sampling', seed=seed[0])
        ax_af.plot(configs_dense, predictions_gb, label="gradient boosting", color='tab:orange')
        ax_af.fill_between(
            configs_dense.flatten(),
            predictions_gb.flatten(),
            np.zeros_like(predictions_gb.flatten()),
            alpha=0.3,
            color='tab:orange'
        )

    ax_func.set_xlim(-0.01, 1.01)
    ax_af.set_ylim(-0.01, 1.01)
    ax_af.set_xlabel(r'$x$')
    ax_af.set_xticks(np.arange(0, 1.1, step=0.2))
    ax_af.set_title("Acuisition function")
    ax_af.axvline(x=configs[-1, :], linestyle='--', color='tab:gray')
    plt.tight_layout()
    plt.savefig(root_dir / f"{title}.png")
