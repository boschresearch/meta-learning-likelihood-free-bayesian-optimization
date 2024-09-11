# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from copy import deepcopy

import torch
import numpy as np
from tqdm.auto import tqdm

from meta_learning.loss import MetaLoss


class EarlyStopping:
    def __init__(self, count_down=20) -> None:
        self.reset_count_down = count_down
        self.count_down = count_down
        self.current_best = np.inf
        self.update_best = False

    def _reset(self):
        self.count_down = self.reset_count_down

    def early_stop(self, current):
        if self.count_down > 0:
            if current < self.current_best:
                self.current_best = current
                self.update_best = True
                self._reset()
            else:
                self.update_best = False
                self.count_down -= 1
            return False
        else:
            return True


class Recorder:
    def __init__(self) -> None:
        self.reset()

    @property
    def loss(self):
        return np.mean(self.loss_history)

    def update_loss(self, loss, **kwargs):
        self.loss_history.append(loss)

    def reset(self):
        self.loss_history = []


def train(
    model,
    data,
    num_epochs,
    batch_size,
    shuffle=True,
    dtype=torch.float64,
    device=torch.device("cpu"),
    train_validation_split=.8,
    *args,
    **kwargs
):
    X, task_ids, y, w = data
    X = torch.tensor(X, dtype=dtype, device=device)
    y = torch.tensor(y, dtype=dtype, device=device)
    w = torch.tensor(w, dtype=dtype, device=device)
    task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)
    tensors = [X, task_ids, y, w]
    dataset = torch.utils.data.TensorDataset(*tensors)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_validation_split, 1 - train_validation_split]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
    )
  
    model.initialize(task_ids[-1], device=device, dtype=dtype)
    training_loop(
        model,
        train_dataloader,
        validation_dataloader,
        num_epochs,
    )


def test_loop(model, dataloader, loss_fn): 
    model.eval()
    with torch.no_grad():
        for _, batch_data in enumerate(dataloader):
            # inputs with x and task_id
            inputs = batch_data[:][:2]
            targets = batch_data[:][2]
            weights = batch_data[:][3]                

            outputs = model(*inputs)
            batch_loss = loss_fn(
                outputs, targets, weights, model.task_embedding.weight
            )

    model.train()
    return batch_loss.detach().cpu().numpy()


def training_loop(
    model,
    train_dataloader,
    validation_dataloader,
    num_epochs,
):  
    early_stopping = EarlyStopping()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.999
    )
    loss_fn = MetaLoss(
        batch_size=train_dataloader.batch_size,
        latent_dim=model.task_embedding_size
    )
    # keep track of history
    train_recorder = Recorder()
    validation_recorder = Recorder()

    model.train()
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            train_recorder.reset()
            validation_recorder.reset()
            for _, batch_data in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)

                # inputs with x and task_id
                inputs = batch_data[:][:2]
                targets = batch_data[:][2]
                weights = batch_data[:][3]            

                outputs = model(*inputs)
                batch_loss = loss_fn(
                    outputs, targets, weights, model.task_embedding.weight[1:]
                )

                batch_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_recorder.update_loss(batch_loss.detach().cpu().numpy())

            validation_loss = test_loop(model, validation_dataloader, loss_fn)
            validation_recorder.update_loss(validation_loss)
            # train_summary_writer.add_scalar('loss', train_recorder.loss, epoch)
            # validation_summary_writer.add_scalar('loss', validation_recorder.loss, epoch)

            scheduler.step()

            if (epoch + 1) % 1 == 0:
                pbar.set_description(f"Epoch {epoch + 1}")
                pbar.set_postfix({
                    "train_loss": f"{train_recorder.loss:.4f}",
                    "validation_loss": f"{validation_recorder.loss:.4f}"
                })

            stop = early_stopping.early_stop(validation_recorder.loss)
            if early_stopping.update_best:
                # cache model
                best_model_state_dict = deepcopy(model.state_dict())
            if stop:
                model.load_state_dict(best_model_state_dict)
                print(f"... Current best loss {early_stopping.current_best:.4f}\n"
                      f"... Early Stopped at epoch {epoch}")
                break

    model.eval()
