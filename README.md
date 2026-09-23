## Run $\mathscr{c}$mHC

```sh
torchrun --standalone --nproc_per_node=8 train_cmhc.py config/train_owt.py config/small_model.py config/with_cmhc_lite.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext --cmhc_gamma=0.97 --cmhc_gamma_num_layers=2
```

* Use Fineweb-Edu: `--dataset=${HUGGINGFACE_HUB_CACHE}/fineweb_edu`
* Change residual streams: `--hyper_conn_n=6`, default to 4
* Change iterations: `--max_iters=20000`, default to 10000
* Change storage directory: `--cmhc_results_dir=<out_dir>/cmhc_results`
* Change hyperparameters: `--cmhc_gamma=0.97 --cmhc_gamma_num_layers=2`. `cmhc_gamma` is $\lambda$, default to 1. `cmhc_gamma_num_layers` is the layers for contraction; default to all layers.
* Change log path: `--log_path=${LOG_PATH}.log`. Default to `./run_summary.log`

## Run baselines

Residual

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py --hyper_conn_type=none --hyper_conn_n=1 --out_prefix_method=residual --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

`HC`

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_hc.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

`mHC`

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

`mHC-lite`

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc_lite.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```



## $\mathscr{c}$mHC operator source code

`ascend-cmhc-core/` provides the AscendC implementations of fused $\mathscr{c}$mHC forward and backward operators for Ascend 910C NPUs.

| Operator | Directory | Description |
|----------|-----------|-------------|
| `MhcPreCmhc` (forward) | `mhc_pre_cmhc/` | Fused forward kernel: RMSNorm → linear projection (`x @ φ + bias`) → sigmoid gating → softmax → permutation-matrix einsum. Produces the branch input `hIn`, gating vector `hPost`, and the doubly-stochastic residual matrix `hRes`. |
| `MhcPreCmhcBackward` (backward) | `mhc_pre_cmhc_backward/` | Backward kernel: computes gradients w.r.t. `x`, `φ` (weight), `α` (alpha), `bias`, and optional `γ` (RMSNorm gamma). |


