"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from hyper_conn.hres_capture import (
    CsvTrainingRecorder,
    HResCaptureMonitor,
    build_hres_heatmap_steps,
    capture_hres_probe_snapshot,
    consume_hres_capture_payload,
    get_hres_capture_modules,
    reset_hres_capture_buffers,
    selected_stream_metric_module_names,
    set_hres_capture_enabled,
)
from run_summary_log import append_run_header, append_run_summary, append_step_summary, format_eval_summary
from pprint import pprint
import warnings
import json
import glob
import random 

# suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# ----- hyper conn start -----
hyper_conn_type = "none" # none, hc, mhc, mhc_lite, kromhc
hyper_conn_n = 1 # num_streams
mhc_init_permutation_indices = 0
capture_hres = False
capture_results_dir = ""
heatmap_points = 4
capture_heatmap_points = -1 # legacy alias; if set, overrides heatmap_points
log_path = "./run_summary.log"

# ----- hyper conn end -----

# I/O
seed = 1337
out_prefix_dataset = ""
out_prefix_model = ""
out_prefix_method = "residual"
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'exp' # 'run' + str(time.time())
wandb_group = "default"
wandb_notes = ""
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
if capture_heatmap_points >= 0:
    heatmap_points = capture_heatmap_points
config = {k: globals()[k] for k in config_keys} # will be useful for logging
run_start_timestamp = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
# -----------------------------------------------------------------------------

MHC_INIT_PERMUTATION_TYPES = {"mhc_lite", "cmhc_lite"}

def apply_mhc_init_permutation_arg(model_args, init_index):
    if model_args["hyper_conn_type"] in MHC_INIT_PERMUTATION_TYPES:
        model_args["mhc_init_permutation_indices"] = init_index
    else:
        model_args.pop("mhc_init_permutation_indices", None)


def config_for_run_log():
    resolved_model_args = globals().get("model_args", {})
    fields = {
        "dataset": dataset,
        "out_dir": out_dir,
        "out_prefix_model": out_prefix_model,
        "out_prefix_method": out_prefix_method,
        "n_layer": resolved_model_args.get("n_layer", n_layer),
        "n_head": resolved_model_args.get("n_head", n_head),
        "n_embd": resolved_model_args.get("n_embd", n_embd),
        "block_size": resolved_model_args.get("block_size", block_size),
        "hyper_conn_type": resolved_model_args.get("hyper_conn_type", hyper_conn_type),
        "hyper_conn_n": resolved_model_args.get("hyper_conn_n", hyper_conn_n),
        "mhc_init_permutation_indices": resolved_model_args.get("mhc_init_permutation_indices", mhc_init_permutation_indices),
        "capture_hres": capture_hres,
        "capture_results_dir": capture_results_dir,
        "heatmap_points": heatmap_points,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "max_iters": max_iters,
    }
    return ", ".join(f"{key}={value}" for key, value in fields.items())

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

random.seed(seed + seed_offset)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(
        project = wandb_project , 
        name    = wandb_run_name + '-' + str(random.randint(1000,9999)), 
        group   = wandb_group,
        config  = config , 
        notes   = wandb_notes,
    )

random.seed(seed + seed_offset)

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Preserve an explicitly provided out_dir from CLI/config; otherwise derive the
# experiment directory name from the configured prefixes.
if out_dir == 'out':
    out_dir = f"out-{out_prefix_dataset}-{out_prefix_model}-{out_prefix_method}"
if capture_results_dir == "":
    capture_results_dir = os.path.join(out_dir, "hres_capture")
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Batch sampling (simple random contiguous windows)




# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
vocab_size = meta_vocab_size

# model init
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    vocab_size=vocab_size, 
    dropout=dropout,
    hyper_conn_n=hyper_conn_n,
    hyper_conn_type=hyper_conn_type,
) # start with model_args from command line
apply_mhc_init_permutation_arg(model_args, mhc_init_permutation_indices)
if master_process:
    print ("="*100)
    for k, v in model_args.items():
        print (f"{k} = {v}")
    print ("="*100)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'hyper_conn_type', 'hyper_conn_n']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    apply_mhc_init_permutation_arg(
        model_args,
        checkpoint_model_args.get("mhc_init_permutation_indices", mhc_init_permutation_indices),
    )
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if hyper_conn_type != "none" or hyper_conn_n != 1:
        raise ValueError("init_from='gpt2*' does not support hyper connections; use init_from='scratch' or resume a matching checkpoint.")
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # unwrap DDP container if needed
hres_modules = get_hres_capture_modules(raw_model)
hres_num_layers = int(getattr(raw_model.config, "n_layer", n_layer))
hres_num_streams = int(getattr(raw_model.config, "hyper_conn_n", hyper_conn_n))
hres_start_step = iter_num
hres_structured_active = len(hres_modules) > 0
hres_capture_active = capture_hres and hres_structured_active
hres_heatmap_steps = build_hres_heatmap_steps(hres_start_step, max_iters, heatmap_points) if hres_structured_active else []
hres_stream_module_names = selected_stream_metric_module_names(hres_modules, hres_num_layers) if hres_num_streams == 4 else set()
hres_csv_recorder = CsvTrainingRecorder(capture_results_dir) if master_process else None
hres_monitor = None
if master_process:
    append_run_header(run_start_timestamp, log_path, config_for_run_log())
if capture_hres and master_process and len(hres_modules) == 0:
    print("H_res capture requested, but no active hyper-connection modules expose H_res.")
if hres_capture_active and master_process:
    hres_monitor = HResCaptureMonitor(
        results_dir = capture_results_dir,
        module_names = [name for name, _ in hres_modules],
        matrix_size = hres_modules[0][1].num_fracs * hres_modules[0][1].num_residual_streams,
        start_step = hres_start_step,
        final_step = max_iters,
        heatmap_points = heatmap_points,
        report_title = f"{hyper_conn_type} H_res Capture Report",
    )
    print(f"H_res capture outputs: {capture_results_dir}")
set_hres_capture_enabled(hres_modules, False)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    if hres_monitor is not None:
        set_hres_capture_enabled(hres_modules, False)
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
train_losses = []
tpss = [] # token per second
grad_norms = []
last_completed_step = iter_num - 1
last_eval_desc = None
last_eval_step = None

if hres_structured_active:
    probe_X, probe_Y = get_batch('train')
    initial_payload = capture_hres_probe_snapshot(model, hres_modules, (probe_X, probe_Y), ctx, stream_module_names = hres_stream_module_names)
    if master_process:
        if hres_csv_recorder is not None:
            hres_csv_recorder.log_hres_snapshot(step = iter_num, label = "initial", module_payloads = initial_payload)
            hres_csv_recorder.log_stream_snapshot(step = iter_num, label = "initial", module_payloads = initial_payload, num_layers = hres_num_layers)
        if hres_monitor is not None:
            hres_monitor.capture_heatmap_snapshot(
                step = iter_num,
                module_payloads = initial_payload,
                label = "initial",
            )
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
        avg_grad_norm = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]

        desc = f"[step {iter_num}]" + ", ".join([
            f"train loss: {losses['train']:.4f}" , 
            f"val loss: {losses['val']:.4f}" , 
            f"(avg train loss: {avg_train_loss[0]:.4f} +/- {avg_train_loss[1]:.4f})" ,
            f"(avg grad norm: {avg_grad_norm[0]:.4f} +/- {avg_grad_norm[1]:.4f})" ,
        ])
        print(desc)
        append_step_summary(run_start_timestamp, desc, log_path)
        if hres_csv_recorder is not None:
            hres_csv_recorder.log_step(
                step = iter_num,
                train_loss = float(losses['train']),
                val_loss = float(losses['val']),
                avg_train_loss = float(avg_train_loss[0]),
                std_train_loss = float(avg_train_loss[1]),
                avg_grad_norm = float(avg_grad_norm[0]),
                std_grad_norm = float(avg_grad_norm[1]),
            )
        last_eval_desc = desc
        last_eval_step = iter_num


        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss_est": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "train/avg_loss": avg_train_loss[0],
                "train/avg_gnorm": avg_grad_norm[0],
            }, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    train_time_start = time.time()
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    if hres_monitor is not None:
        reset_hres_capture_buffers(hres_modules)
        set_hres_capture_enabled(hres_modules, True)
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    hres_step_payload = consume_hres_capture_payload(hres_modules) if hres_monitor is not None else None
    if hres_monitor is not None:
        set_hres_capture_enabled(hres_modules, False)
    # clip the gradient
    grad_norm = -1.
    if grad_clip > 0.:
        scaler.unscale_(optimizer)
        ret_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(ret_grad_norm)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    train_time_end = time.time()
    d_train_time = train_time_end - train_time_start
    tokens_per_sec = tokens_per_iter / d_train_time
    
    tpss.append(tokens_per_sec)
    train_losses.append(float(loss))
    grad_norms.append(float(grad_norm))
    if len(train_losses) > 200:
        train_losses.pop(0)
    if len(grad_norms) > 200:
        grad_norms.pop(0)

    if master_process and hres_monitor is not None:
        hres_monitor.log_training_step(
            step = iter_num,
            train_loss = float(loss.item() * gradient_accumulation_steps),
            grad_norm = grad_norm,
            module_payloads = hres_step_payload,
        )
    if master_process and hres_csv_recorder is not None:
        hres_csv_recorder.log_iter(
            step = iter_num,
            train_loss = float(loss.item() * gradient_accumulation_steps),
            grad_norm = grad_norm,
            lr = lr,
            throughput = tokens_per_sec,
        )
    if hres_structured_active and iter_num in hres_heatmap_steps[1:-1]:
        probe_payload = capture_hres_probe_snapshot(model, hres_modules, (probe_X, probe_Y), ctx, stream_module_names = hres_stream_module_names)
        if master_process:
            if hres_csv_recorder is not None:
                hres_csv_recorder.log_hres_snapshot(step = iter_num, label = "sample", module_payloads = probe_payload)
                hres_csv_recorder.log_stream_snapshot(step = iter_num, label = "sample", module_payloads = probe_payload, num_layers = hres_num_layers)
            if hres_monitor is not None:
                hres_monitor.capture_heatmap_snapshot(
                    step = iter_num,
                    module_payloads = probe_payload,
                    label = "sample",
                )


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        tokens_seen = iter_num * tokens_per_iter

        desc = f"[iter {iter_num}]" + ", ".join([
            f"loss: {lossf:.4f}" , 
            f"tokens/sec: {np.mean(tpss):.2f} +/- {np.std(tpss):.2f}" if len(tpss) > 0 else "0.00 +/- 0.00" ,
            f"tokens seen: {tokens_seen}" , 
            f"grad norm: {grad_norm:.4f}" , 
        ])
        if wandb_log:
            wandb.log({
                "train/loss": lossf,
                "train/tokens_per_sec": np.mean(tpss) if len(tpss) > 0 else 0,
                "train/tokens_seen": tokens_seen,
                "train/grad_norm": grad_norm,
            }, step=iter_num)

        if hres_monitor is not None and wandb_log:
            spectral_norms = np.array(
                [hres_step_payload[name]["spectral_norm"] for name, _ in hres_modules],
                dtype = np.float32,
            )
            wandb.log({
                "hres/spectral_norm_mean": float(spectral_norms.mean()),
                "hres/spectral_norm_max": float(spectral_norms.max()),
            }, step=iter_num)

        print(desc)
    last_completed_step = iter_num
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if hres_structured_active:
    final_step = max(last_completed_step, hres_start_step)
    final_payload = capture_hres_probe_snapshot(model, hres_modules, (probe_X, probe_Y), ctx, stream_module_names = hres_stream_module_names)
    if master_process:
        if hres_csv_recorder is not None:
            hres_csv_recorder.log_hres_snapshot(step = final_step, label = "final", module_payloads = final_payload)
            hres_csv_recorder.log_stream_snapshot(step = final_step, label = "final", module_payloads = final_payload, num_layers = hres_num_layers)
        if hres_monitor is not None:
            hres_monitor.capture_heatmap_snapshot(
                step = final_step,
                module_payloads = final_payload,
                label = "final",
            )
            hres_monitor.finalize()

if master_process:
    final_eval_step = max(last_completed_step, 0)
    final_eval_desc = last_eval_desc
    if final_eval_desc is None or last_eval_step != final_eval_step:
        final_losses = estimate_loss()
        final_eval_desc = format_eval_summary(final_eval_step, final_losses, train_losses, grad_norms)
        print(final_eval_desc)
        append_step_summary(run_start_timestamp, final_eval_desc, log_path)
        if hres_csv_recorder is not None:
            avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
            avg_grad_norm = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]
            hres_csv_recorder.log_step(
                step = final_eval_step,
                train_loss = float(final_losses['train']),
                val_loss = float(final_losses['val']),
                avg_train_loss = float(avg_train_loss[0]),
                std_train_loss = float(avg_train_loss[1]),
                avg_grad_norm = float(avg_grad_norm[0]),
                std_grad_norm = float(avg_grad_norm[1]),
                label = "final",
            )
    append_run_summary(run_start_timestamp, final_eval_desc, log_path)

if ddp:
    destroy_process_group()
