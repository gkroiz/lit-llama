"""
This script is a placeholder for training LLaMA from scratch.
Currently, it just trains on the Shakespeare dataset.
"""

import os
import csv
import time
from functools import partial
from typing import Tuple

import lightning as L
from lightning.fabric.strategies import XLAStrategy, XLAFSDPStrategy

import torch

from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch_xla.core.xla_model as xm


import numpy as np

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.utils import save_model_checkpoint

track_times = True
strategy = 'fsdp' #or 'fsdp'
out_dir = "out/training"
eval_interval = 2000
eval_iters = 200
log_interval = 1
# compilation fails as it does not support torch.complex64 for RoPE
# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 4
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# For shakespeare, choose smaller block size than vanilla LLaMA
block_size = 1024


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    iter_num = 0

    iter_times = []
    get_batch_times = []
    foward_times = []
    loss_times = []
    backward_times = []
    optimizer_step_times = []
    zero_grad_times = []

    if track_times and xm.get_ordinal() == 0:
        f = open('times.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(['iter', 'iter_times', 'get_batch_times', 'forward_times', 'loss_times', 'backward_times', 'opt_step_times', 'zero_grad_times'])

    # close the file
    while True:
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        t0 = time.time()
        get_batch_start = time.time()
        input_ids, targets = get_batch(
            fabric,
            train_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        get_batch_end = time.time()

        foward_start = time.time()
        logits = model(input_ids)
        foward_end = time.time()

        loss_start = time.time()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        loss_end = time.time()

        backward_start = time.time()
        fabric.backward(loss)
        backward_end = time.time()

        # TODO: Gradient clipping
        # if grad_clip != 0.0:
        #     fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

        optimizer_step_start = time.time()
        if isinstance(fabric._strategy, XLAFSDPStrategy):
            optimizer.step()
            xm.mark_step()
        elif isinstance(fabric._strategy, XLAStrategy):
            xm.optimizer_step(optimizer, barrier=True)
        else:
            raise Exception("Skipped optimizer step!")
        optimizer_step_end = time.time()

        zero_grad_start = time.time()
        optimizer.zero_grad()
        zero_grad_end = time.time()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            if track_times and xm.get_ordinal() == 0:
                iter_times.append(dt)
                get_batch_times.append(get_batch_end-get_batch_start)
                foward_times.append(foward_end-foward_start)
                loss_times.append(loss_end-loss_start)
                backward_times.append(backward_end-backward_start)
                optimizer_step_times.append(optimizer_step_end-optimizer_step_start)
                zero_grad_times.append(zero_grad_end-zero_grad_start)

            # fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        iter_num += 1

        if iter_num > max_iters:
            break

    if track_times and xm.get_ordinal() == 0:
        for i in range(len(iter_times)):
            writer.writerow([i, iter_times[i], get_batch_times[i], foward_times[i], loss_times[i], backward_times[i], optimizer_step_times[i], zero_grad_times[i]])
        f.close()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(
            fabric,
            val_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x = x.to(xm.xla_device())
    y = y.to(xm.xla_device())
    return x, y


def load_datasets(data_dir: str = "data/shakespeare") -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


def launch_func(fabric):
    fabric.seed_everything(1337)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()
    config = LLaMAConfig.from_name("test")
    config.block_size = block_size
    config.vocab_size = 100  # from prepare_shakespeare.py

    with fabric.device:
        model = LLaMA(config)

    # if compile:
    #     model = torch.compile(model)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    if isinstance(fabric._strategy, XLAFSDPStrategy):
        model = fabric.setup(model)
        optimizer = fabric.setup_optimizers(optimizer)
    elif isinstance(fabric._strategy, XLAStrategy):
        model, optimizer = fabric.setup(model, optimizer)
    else:
        raise Exception("Please select either xla or fsdp on TPUs")

    fabric.print('Beginning training')
    train(fabric, model, optimizer, train_data, val_data)


def main() -> None:
    global strategy
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    if strategy == 'xla':
        strategy = XLAStrategy(broadcast_master_params=False)
    elif strategy == 'fsdp' or strategy == 'xla_fsdp':
        strategy = XLAFSDPStrategy(auto_wrap_policy=auto_wrap_policy, broadcast_master_params=False)
    else:
        raise Exception("Please select either xla or fsdp on TPUs")

    fabric = L.Fabric(accelerator="tpu", devices=4, strategy=strategy)
    fabric.launch(launch_func)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()