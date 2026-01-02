from network_util import *
from training_loop_util import *
from tqdm import tqdm
import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime


# 各种常量
# 打开文件
TRAIN_DATA = "data/TinyStory-train-uint16.raw"
VALID_DATA = "data/TinyStory-valid-uint16.raw"
DATA_TYPE = np.uint16

# 保存文件（统一前缀）
SAVE_PREFIX = "ts_model"

# 训练超参数
# data
EPOCH_NUM = 25
BATCH_NUM = 1024
BATCH_SIZE = 256

# lr scheduler
MAX_LR = 5e-5
MIN_LR = 1e-6
WARMUP = int(EPOCH_NUM * BATCH_NUM * 0.03)
COSINE_CYCLE = EPOCH_NUM * BATCH_NUM - WARMUP

# grad clip
MAX_NORM = 5.0

# AdamW
BETAS = (0.9, 0.95)
DECAY = 0.0001

# 配置模型
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
EMBEDDING_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 16
FFN_DIM = 1344
ROPE_THETA = 10000


# 需要定义一些简单的函数，来实现功能
# lr schedule
def set_lr(
    opt: torch.optim.Optimizer,
    it: int,
):
    new_lr = lr_schedule(it, MAX_LR, MIN_LR, WARMUP, COSINE_CYCLE)
    for group in opt.param_groups:
        group["lr"] = new_lr

    return new_lr


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入data
# dataset = np.memmap(TRAIN_DATA, dtype=DATA_TYPE, mode="r")
# valid_dataset = np.memmap(VALID_DATA, dtype=DATA_TYPE, mode="r")

dataset = np.fromfile(TRAIN_DATA, dtype=DATA_TYPE)
valid_dataset = np.fromfile(VALID_DATA, dtype=DATA_TYPE)

# 数据搬运优化
dataset = torch.from_numpy(dataset).to(torch.long).pin_memory()
valid_dataset = torch.from_numpy(valid_dataset).to(torch.long).pin_memory()


model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=FFN_DIM,
    rope_theta=ROPE_THETA,
    device=device,
)

model.to(device)

loss_fn = crossEntropy(device)
optimizer = AdamW(model.parameters(), MAX_LR, BETAS, DECAY, 1e-8)

# 训练开始前 打印日志


def print_and_log(msg: str, log_path: str | None = None):
    print(msg)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def dump_hparams(log_dir: str, hparams: dict):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "hparams.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    return path


# ========== 在训练前设置一个 log_dir ==========
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(SAVE_PREFIX if SAVE_PREFIX else ".", f"run_{run_id}")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "train.log")


steps_per_epoch = BATCH_NUM
total_steps = EPOCH_NUM * steps_per_epoch

hparams = {
    "EPOCH_NUM": EPOCH_NUM,
    "BATCH_NUM": BATCH_NUM,
    "BATCH_SIZE": BATCH_SIZE,
    "VOCAB_SIZE": VOCAB_SIZE,
    "CONTEXT_LENGTH": CONTEXT_LENGTH,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "NUM_LAYERS": NUM_LAYERS,
    "NUM_HEADS": NUM_HEADS,
    "FFN_DIM": FFN_DIM,
    "ROPE_THETA": ROPE_THETA,
    "MAX_LR": MAX_LR,
    "MIN_LR": MIN_LR,
    "WARMUP": WARMUP,
    "COSINE_CYCLE": COSINE_CYCLE,
    "MAX_NORM": MAX_NORM,
    "BETAS": BETAS,
    "DECAY": DECAY,
    "device": str(device),
    "steps_per_epoch": steps_per_epoch,
    "total_steps": total_steps,
    "TRAIN_DATA": TRAIN_DATA,
    "VALID_DATA": VALID_DATA,
}

hp_path = dump_hparams(log_dir, hparams)
print_and_log(f"[run_id] {run_id}", log_path)
print_and_log(f"[hparams] saved to: {hp_path}", log_path)
print_and_log(f"[device] {device}", log_path)
print_and_log(
    f"[steps] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}", log_path
)


# 训练loop
iter_num = 0
all_loss = []
all_valid_loss = []
for epoch in range(EPOCH_NUM):
    model.train()
    epoch_loss = 0.0
    data_sampler = EpochSampler(len(dataset) - CONTEXT_LENGTH)
    for batch in tqdm(range(BATCH_NUM), desc="Training in batch"):
        # for batch in range(BATCH_NUM):
        iter_num += 1
        inputs, labels = get_batch_without_same(
            dataset, BATCH_SIZE, CONTEXT_LENGTH, data_sampler, device
        )
        optimizer.zero_grad()
        cur_lr = set_lr(optimizer, iter_num)

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        grad_clipping(model.parameters(), MAX_NORM)

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / BATCH_NUM

    time_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_and_log(
        f"[{time_info}] Epoch [{epoch+1}/{EPOCH_NUM}] lr={cur_lr:.6e} train_loss={avg_loss:.4f}",
        log_path,
    )
    all_loss.append(avg_loss)

    # 验证部分
    with torch.no_grad():
        model.eval()

        valid_loss = 0.0
        valid_sample_num = BATCH_NUM // 16
        valid_sampler = EpochSampler(len(valid_dataset) - CONTEXT_LENGTH)
        for batch in range(valid_sample_num):
            inputs, labels = get_batch_without_same(
                valid_dataset, BATCH_SIZE, CONTEXT_LENGTH, valid_sampler, device
            )
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()

        avg_loss = valid_loss / valid_sample_num
        all_valid_loss.append(avg_loss)

        save_model_name = SAVE_PREFIX + "_" + run_id + f"_at_{epoch + 1}.pt"
        model_path = os.path.join(log_dir, save_model_name)

        # 保存一下模型
        save_checkpoint(model, optimizer, iter_num, model_path)
        time_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print_and_log(
            f"[{time_info}] Epoch [{epoch+1}/{EPOCH_NUM}] valid_loss={avg_loss:.4f}",
            log_path,
        )


def plot_losses(train_losses, valid_losses, save_path: str):
    plt.figure()
    plt.plot(train_losses, label="train")
    if valid_losses is not None and len(valid_losses) > 0:
        plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# 训练结束后：
plot_path = os.path.join(log_dir, "loss_curve.png")
plot_losses(all_loss, all_valid_loss, plot_path)
print_and_log(f"[plot] saved to: {plot_path}", log_path)
