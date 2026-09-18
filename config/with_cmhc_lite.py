hyper_conn_type = "cmhc_lite"
hyper_conn_n = 4

# Use `0.0 <= cmhc_gamma < 1.0` to enable the shrinkage.
# `cmhc_gamma = 1.0` keeps the original mHC-lite permutation basis.
cmhc_gamma = 1.0

# `all` counts over every H_res module in model order.
# `attn` / `mlp` count only over the attention / MLP H_res subset.
# `layer` preserves the old transformer-block counting semantics.
cmhc_gamma_scope = "all"

# `"all"` applies the configured gamma to every H_res selected by `cmhc_gamma_scope`.
# `0` disables shrinkage on every selected H_res.
# positive `k` applies gamma to the first `k` selected H_res modules.
# negative `-k` applies gamma to the last `k` selected H_res modules.
# when `cmhc_gamma_scope = "layer"`, integers keep the legacy last-`k` transformer-block semantics.
cmhc_gamma_num_layers = "all"

# `all` counts over every H_res module in model order.
# `attn` / `mlp` count only over the attention / MLP H_res subset.
cmhc_identity_scope = "all"

# `0` keeps every selected H_res trainable.
# positive `k` fixes the first `k` selected H_res modules to identity.
# negative `-k` fixes the last `k` selected H_res modules to identity.
cmhc_identity_num_hres = 0

# Leave empty to write under `<out_dir>/cmhc_results`.
cmhc_results_dir = ""

# Number of uniformly sampled H_res heatmap checkpoints, including start and end.
heatmap_points = 4

wandb_notes = "cmhc-lite"
out_prefix_method = "cmhc-lite"
