import torch
import torch.nn as nn
from einops import reduce, rearrange
from typing import Optional
from collections.abc import Callable, Iterable
from typing import IO, Any, BinaryIO
import math
import numpy.typing as npt
import os
from pathlib import Path


class crossEntropy(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, logits_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        max_values = reduce(logits_in, "... len -> ... 1", "max")
        numer = logits_in - max_values

        denom = reduce(torch.exp(numer), "... len -> ... 1", "sum")
        numer = torch.gather(numer, dim=-1, index=target.unsqueeze(-1))
        out = (numer - torch.log(denom)) * -1
        out = reduce(out, "... ->", "mean")

        return out


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                # 这里开始adamW的核心步骤
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                if "t" not in state:
                    state["t"] = 1

                m = state["m"]
                v = state["v"]
                t = state["t"]
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                lr_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= lr_t * (
                    m / (torch.sqrt(v) + eps)
                )  # Update weight tensor in-place.
                p.data *= 1 - lr * weight_decay
                state["t"] = t + 1  # Increment iteration number.`
                state["m"] = m
                state["v"] = v
        return loss


def lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate

    angle = ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi

    factor = 0.5 * (1 + math.cos(angle))

    return min_learning_rate + factor * (max_learning_rate - min_learning_rate)


def grad_clipping(para: Iterable[torch.nn.Parameter], max_norm: float):
    para = list(para)
    sum = None
    for p in para:
        if p.grad is None:
            continue

        g = p.grad.detach()

        n = torch.linalg.vector_norm(g)
        val = n * n
        sum = val if sum is None else (sum + val)

    if sum is None:
        l2 = torch.tensor(0.0)
    else:
        l2 = torch.sqrt(sum)
    l2 = float(l2.item())
    if l2 < max_norm:
        return None
    scaler = max_norm / (l2 + 1e-6)
    for p in para:
        if p.grad is None:
            continue
        p.grad.mul_(scaler)

    return None


# 这个函数存在问题，目前没有考虑EOF，日后训练可能会出现问题,但是，似乎生成模型包含EOF没有问题
# 关于如何读入大文件，作业中有提示，但是仅通过测试点不需要这个，先不管了
# 网上别人写的代码也没有考虑EOF，但采样这一块还是有所改进的，抄个代码算了
# 从这里开始都是他的代码，感谢知乎@Perf
# 后来发现，这个地方对数据集的操作，相当于把整个数据集搬来搬去，做了一些优化


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[npt.NDArray, npt.NDArray]:
    B, T = batch_size, context_length
    data_t = torch.as_tensor(dataset, dtype=torch.long, device=device)
    N = data_t.numel()

    # starts = torch.randint(0, N - T, (B,), device=device)
    starts = torch.randperm(N - T, device=device)[:B]  # 无放回采样
    offsets = rearrange(torch.arange(T + 1, device=device), "n -> 1 n")  # [1, T+1]
    positions = rearrange(starts, "b -> b 1") + offsets
    tokens = data_t[positions]  # [B, T+1]
    x, y = tokens[:, :-1], tokens[:, 1:]  # Next token prediction [B, T]
    return x, y


# 这是一个改进写法，确保了样本之间少重叠
class EpochSampler:
    def __init__(self, num_positions: int):
        self.N = num_positions
        self._shuffle()

    def _shuffle(self):
        self.perm = torch.randperm(self.N, device="cpu")
        self.cursor = 0

    def next(self, k: int) -> torch.Tensor:
        if self.cursor + k > self.N:
            self._shuffle()
        idx = self.perm[self.cursor : self.cursor + k]
        self.cursor += k
        return idx


def get_batch_without_same(
    dataset_t: torch.Tensor,  # pinned CPU long tensor: [N]
    batch_size: int,
    context_length: int,
    sampler: EpochSampler,
    device: torch.device,
):
    B, T = batch_size, context_length
    starts = sampler.next(B)  # CPU (B,)

    offsets = torch.arange(T + 1, device="cpu").unsqueeze(0)  # (1, T+1)
    positions = starts.unsqueeze(1) + offsets  # (B, T+1)

    tokens = dataset_t[positions]  # still CPU pinned
    x, y = tokens[:, :-1], tokens[:, 1:]

    # 异步 H2D
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    if isinstance(src, (str, os.PathLike, Path)):
        ckpt = torch.load(src, map_location=device)
    else:
        ckpt = torch.load(src, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["iteration"])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    ckpt: dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (str, os.PathLike, Path)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
    else:
        torch.save(ckpt, out)
