from model import GPT
from config import GPTConfig, TrainingConfig
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import Task

train_config = TrainingConfig()
out_dir = "out/"
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
resume = False

tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * GPTConfig.block_size
)
print("Tokens per iteration: ", tokens_per_iter)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

ctx = torch.autocast(train_config.device, dtype=torch.bfloat16)

model_args = dict(
    n_layer=GPTConfig.n_layer,
    n_head=GPTConfig.n_head,
    n_embed=GPTConfig.n_embed,
    block_size=GPTConfig.block_size,
    bias=GPTConfig.bias,
    vocab_size=GPTConfig.vocab_size,
    dropout=GPTConfig.dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=GPTConfig.block_size,
    device=train_config.device,
    num_workers=0,
    max_shards=1
)

best_val_loss = 1e9
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(train_config.device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
)

if train_config.compile:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()
while True:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
        )
        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            print(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % train_config.log_interval == 0:
        lossf = loss.item() * train_config.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num > train_config.max_iters:
        break

writer.close()
