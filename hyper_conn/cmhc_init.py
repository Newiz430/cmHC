from .hyper_connections import (
    get_init_and_expand_reduce_stream_functions
)

from .mhc import (
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions
)

from .mhc_lite import (
    get_init_and_expand_reduce_stream_functions as mhclite_get_init_and_expand_reduce_stream_functions
)

from .mhc_analysis import (
    get_init_and_expand_reduce_stream_functions as mhc_analysis_get_init_and_expand_reduce_stream_functions
)

from .cmhc_lite import (
    get_init_and_expand_reduce_stream_functions as cmhclite_get_init_and_expand_reduce_stream_functions
)

from .kromhc import (
    get_init_and_expand_reduce_stream_functions as kromhc_get_init_and_expand_reduce_stream_functions
)

last_announced_key = None


def resolve_cmhc_resume_scopes(
    checkpoint_model_args,
    *,
    default_gamma_scope = "all",
    default_identity_scope = "all",
):
    gamma_scope = checkpoint_model_args.get("cmhc_gamma_scope")
    if (
        gamma_scope is None and
        checkpoint_model_args.get("hyper_conn_type") == "cmhc_lite" and
        "cmhc_gamma_num_layers" in checkpoint_model_args
    ):
        gamma_scope = "layer"
    gamma_scope = default_gamma_scope if gamma_scope is None else gamma_scope

    identity_scope = checkpoint_model_args.get("cmhc_identity_scope", default_identity_scope)
    return gamma_scope, identity_scope

def _announce_hyper_conn(config):
    message = f"HYPER_CONN: USING {config.hyper_conn_type} with {config.hyper_conn_n} streams"
    try:
        from megatron.training import print_rank_0
    except Exception:
        print(message)
        return
    print_rank_0(message)

def reset_hyper_conn_announce_state():
    global last_announced_key
    last_announced_key = None

def hyper_conn_init_func(config):
    global last_announced_key

    config_key = (
        config.hyper_conn_type,
        getattr(config, "hyper_conn_n", None),
        getattr(config, "cmhc_gamma", None),
        getattr(config, "cmhc_gamma_num_layers", None),
        getattr(config, "cmhc_gamma_scope", None),
        getattr(config, "cmhc_identity_num_hres", None),
        getattr(config, "cmhc_identity_scope", None),
        getattr(config, "mhc_init_permutation_indices", None),
    )
    if config.hyper_conn_type != "none" and config_key != last_announced_key:
        _announce_hyper_conn(config)
        last_announced_key = config_key

    if config.hyper_conn_type == "none":
        return get_init_and_expand_reduce_stream_functions(config.hyper_conn_n, disable = True)
    elif config.hyper_conn_type == "hc":
        return get_init_and_expand_reduce_stream_functions(config.hyper_conn_n)
    elif config.hyper_conn_type == "mhc":
        return mc_get_init_and_expand_reduce_stream_functions(config.hyper_conn_n)
    elif config.hyper_conn_type == "mhc_lite":
        return mhclite_get_init_and_expand_reduce_stream_functions(
            config.hyper_conn_n,
            mhc_init_permutation_indices = getattr(config, "mhc_init_permutation_indices", 0),
        )
    elif config.hyper_conn_type == "cmhc_lite":
        return cmhclite_get_init_and_expand_reduce_stream_functions(
            config.hyper_conn_n,
            cmhc_gamma = config.cmhc_gamma,
            cmhc_gamma_num_layers = config.cmhc_gamma_num_layers,
            cmhc_gamma_scope = getattr(config, "cmhc_gamma_scope", "all"),
            cmhc_identity_num_hres = config.cmhc_identity_num_hres,
            cmhc_identity_scope = getattr(config, "cmhc_identity_scope", "all"),
            mhc_init_permutation_indices = getattr(config, "mhc_init_permutation_indices", 0),
            num_model_layers = config.n_layer,
        )
    elif config.hyper_conn_type == "kromhc":
        return kromhc_get_init_and_expand_reduce_stream_functions(
            config.hyper_conn_n,
            mhc_init_permutation_indices = getattr(config, "mhc_init_permutation_indices", None),
        )
    elif config.hyper_conn_type == "analysis":
        return mhc_analysis_get_init_and_expand_reduce_stream_functions(config.hyper_conn_n)
    else:
        raise ValueError(f"Invalid hyper connection type: {config.hyper_conn_type}")
