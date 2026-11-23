# cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction

<div align="center">

[![Github](https://img.shields.io/badge/GitHub-repo-blue?style=for-the-badge&logo=github)](https://github.com/Newiz430/cmHC)
[![Paper](https://img.shields.io/badge/Paper-2611.27193-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.27193)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Papers-%231%20Paper%20of%20the%20day-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/papers/2611.27193)

</div>

Implementation of [cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction](https://arxiv.org/abs/2611.27193), an exact doubly stochastic framework which geometrically contracts the Birkhoff polytope toward its interior. 

This contraction mitigates under-mixing by forcing the residual matrices to explore deeper into the polytope, with a provable stream mixing bound and no new parameterization. Moreover, we apply a moderate contraction at the shallow layers to prevent over-mixing, enabling stable gradient propagation on extremely large-scale model pretraining. 

## Run cmHC

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

[`HC`](https://github.com/lucidrains/hyper-connections)

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_hc.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

[`mHC`](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

[`mHC-lite`](https://github.com/FFTYYY/mhc-lite)

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc_lite.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
```

[`KromHC`](https://github.com/wz1119/KromHC)

~~~sh
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_kromhc.py --dataset=${HUGGINGFACE_HUB_CACHE}/openwebtext
~~~

## cmHC operator source code

`ascend-cmhc-core/` provides the AscendC implementations of fused cmHC forward and backward operators for Ascend 910C NPUs.

| Operator | Directory | Description |
|----------|-----------|-------------|
| `MhcPreCmhc` (forward) | `mhc_pre_cmhc/` | Fused forward kernel: RMSNorm → linear projection (`x @ φ + bias`) → sigmoid gating → softmax → permutation-matrix einsum. Produces the branch input `hIn`, gating vector `hPost`, and the doubly-stochastic residual matrix `hRes`. |
| `MhcPreCmhcBackward` (backward) | `mhc_pre_cmhc_backward/` | Backward kernel: computes gradients w.r.t. `x`, `φ` (weight), `α` (alpha), `bias`, and optional `γ` (RMSNorm gamma). |

## Acknowledgements

cmHC is built mainly upon [nanoGPT](https://github.com/karpathy/nanoGPT), [mHC-lite](https://github.com/FFTYYY/mhc-lite), and other Hyper-Connections baselines, under the [Apache 2.0 license](LICENSE). 

## Citation

Please consider citing our work if it helps:

```bibtex
@article{cmhc,
  title={cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction}, 
  author={He, Haowei and Zhao, Ziwen and Pan, Chenshuo and Lin, Gang and Yang, Zhihao and Wang, Chao and Liu, Xinzhang and Yu, Kaidong and Gao, Han and Li, Dongfang and Yang, Menglin and Song, Shuangyong and Li, Yongxiang},
  journal={arXiv:2611.27193},
  year={2026}
}
```