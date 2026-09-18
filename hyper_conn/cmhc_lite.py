from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce
import itertools
from .hres_capture import (
    consume_hres_capture_state as _consume_hres_capture_state,
    init_hres_capture_state as _init_hres_capture_state,
    record_hres_capture_state as _record_hres_capture_state,
    record_stream_capture_state as _record_stream_capture_state,
    reset_hres_capture_state as _reset_hres_capture_state,
    set_hres_capture as _set_hres_capture,
    set_hres_stream_capture as _set_hres_stream_capture,
)

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y


VALID_CMHC_GAMMA_SCOPES = {"all", "attn", "mlp", "layer"}
VALID_CMHC_IDENTITY_SCOPES = {"all", "attn", "mlp"}
VALID_HRES_KINDS = {"attn", "mlp"}


def normalize_cmhc_gamma_scope(scope):
    scope = default(scope, "all")
    scope = str(scope).lower()
    assert scope in VALID_CMHC_GAMMA_SCOPES, f"`cmhc_gamma_scope` must be one of {sorted(VALID_CMHC_GAMMA_SCOPES)}"
    return scope


def normalize_cmhc_identity_scope(scope):
    scope = default(scope, "all")
    scope = str(scope).lower()
    assert scope in VALID_CMHC_IDENTITY_SCOPES, f"`cmhc_identity_scope` must be one of {sorted(VALID_CMHC_IDENTITY_SCOPES)}"
    return scope


def cmhc_gamma_selects_all(value):
    return isinstance(value, str) and value.lower() == "all"


def normalize_hres_kind(kind):
    if not exists(kind):
        return None

    kind = str(kind).lower()
    assert kind in VALID_HRES_KINDS, f"`hres_kind` must be one of {sorted(VALID_HRES_KINDS)}"
    return kind

# sinkhorn

def l1norm(t, dim):
    return F.normalize(t, p = 1, dim = dim)


def get_all_permutations(
    n: int,
    *,
    gamma: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None
):
    assert n >= 1, "n should be larger than 0"

    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long, device=device)

    eye = torch.eye(n, dtype=dtype, device=device)
    perm_mats = eye[index]  # (n!, n, n)

    if gamma < 1.:
        perm_mats = gamma * perm_mats + (1. - gamma) / n

    return perm_mats


def normalize_init_permutation_index(index, num_permutations: int) -> int:
    assert not isinstance(index, bool), "`mhc_init_permutation_indices` must be an integer"
    try:
        init_index = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError("`mhc_init_permutation_indices` must be an integer") from exc
    assert 0 <= init_index < num_permutations, (
        f"`mhc_init_permutation_indices` must be in [0, {num_permutations - 1}], got {init_index}"
    )
    return init_index

def get_cached_permutations(
    n: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    gamma: float
):
    cache_key = (n, str(device), str(dtype), float(gamma))

    if cache_key not in perm_mats:
        perm_mats[cache_key] = get_all_permutations(
            n,
            gamma = gamma,
            dtype = dtype,
            device = device
        )

    return perm_mats[cache_key]

# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)

    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    cmhc_gamma = 1.,
    cmhc_gamma_num_layers = "all",
    cmhc_gamma_scope = "all",
    cmhc_identity_num_hres = 0,
    cmhc_identity_scope = "all",
    mhc_init_permutation_indices = 0,
    num_model_layers = None,
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = CMHCLite if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        num_fracs = num_fracs,
        cmhc_gamma = cmhc_gamma,
        cmhc_gamma_num_layers = cmhc_gamma_num_layers,
        cmhc_gamma_scope = cmhc_gamma_scope,
        cmhc_identity_num_hres = cmhc_identity_num_hres,
        cmhc_identity_scope = cmhc_identity_scope,
        mhc_init_permutation_indices = mhc_init_permutation_indices,
        num_model_layers = num_model_layers,
        **kwargs
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection residual streams

perm_mats = {}

class CMHCLite(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        num_model_layers = None,
        hres_index = None,
        num_hres = None,
        hres_kind = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual = True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views = 1,                # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn = add,
        num_fracs = 1,                      # https://arxiv.org/abs/2503.14125
        cmhc_gamma = 1.,
        cmhc_gamma_num_layers = "all",
        cmhc_gamma_scope = "all",
        cmhc_identity_num_hres = 0,
        cmhc_identity_scope = "all",
        mhc_init_permutation_indices = 0,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch

        assert 0. <= cmhc_gamma <= 1., '`cmhc_gamma` must be in [0, 1]'

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'

        dim //= num_fracs # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        # self.norm = RMSNorm(dim * num_residual_streams * num_fracs)

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index
        self.num_model_layers = num_model_layers
        self.hres_index = hres_index
        self.num_hres = num_hres
        self.hres_kind = normalize_hres_kind(hres_kind)
        self.cmhc_gamma = cmhc_gamma
        self.cmhc_gamma_num_layers = cmhc_gamma_num_layers
        self.cmhc_gamma_scope = normalize_cmhc_gamma_scope(cmhc_gamma_scope)
        self.cmhc_identity_num_hres = int(cmhc_identity_num_hres)
        self.cmhc_identity_scope = normalize_cmhc_identity_scope(cmhc_identity_scope)
        self.mhc_init_permutation_indices = None
        init_residual_index = randrange(num_residual_streams) % num_residual_streams # match mhc-lite init while reserving layer metadata for cmhc-only features

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams
        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection
        # ------
        # XXX MHC Lite impl. 
        # H_res is from nC to n!
        # ------
        perms = get_cached_permutations(
            num_residual_streams,
            device = "cpu",
            dtype = torch.float32,
            gamma = self.get_effective_gamma()
        )
        self.mhc_init_permutation_indices = normalize_init_permutation_index(
            mhc_init_permutation_indices,
            len(perms),
        )
        self.use_identity_hres = self.should_use_identity_hres()

        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.
        if self.use_identity_hres:
            self.static_alpha_pre = nn.Parameter(init_alpha0.view(-1))
            self.dynamic_alpha_pre_fn = nn.Parameter(
                torch.zeros(
                    dim * num_residual_streams,
                    num_fracs * num_residual_streams * num_input_views
                )
            )
        else:
            init_alpha1 = torch.ones(len(perms) * num_fracs) * -8
            init_alpha1[self.mhc_init_permutation_indices] = 0.

            # (s*v + s!)
            self.static_alpha = nn.Parameter(cat([
                init_alpha0.view(-1),
                init_alpha1
            ], dim = -1))

            self.dynamic_alpha_fn = nn.Parameter(
                torch.zeros(
                    dim * num_residual_streams,
                    num_fracs * ( len(perms) + num_residual_streams * num_input_views )
                )
            )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        if not self.use_identity_hres:
            self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.
            beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)

            # ------ 
            # XXX
            # same here
            # ------
            dynamic_beta_shape = (
                dim * num_residual_streams,
                num_fracs * num_residual_streams
            ) # preserve backwards compat
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn
        _init_hres_capture_state(self)

    def should_use_identity_hres(self):
        count = int(self.cmhc_identity_num_hres)

        if count == 0:
            return False

        scoped_position = self.get_scoped_hres_position(self.cmhc_identity_scope)
        if scoped_position is None:
            return False

        scoped_index, scoped_total = scoped_position
        count = min(abs(count), scoped_total)

        if self.cmhc_identity_num_hres > 0:
            return scoped_index < count

        return scoped_index >= (scoped_total - count)

    def get_scoped_hres_position(self, scope: str):
        assert self.hres_index is not None and self.num_hres is not None, '`hres_index` and `num_hres` must be passed when `cmhc` scope selection is enabled'

        scope = normalize_cmhc_identity_scope(scope)
        total_hres = int(self.num_hres)
        hres_index = int(self.hres_index)

        if scope == "all":
            return (hres_index, total_hres)

        assert self.hres_kind is not None, '`hres_kind` must be passed when `cmhc` scope is `attn` or `mlp`'

        if self.hres_kind != scope:
            return None

        if scope == "attn":
            scoped_total = (total_hres + 1) // 2
            scoped_index = hres_index // 2
        else:
            scoped_total = total_hres // 2
            scoped_index = (hres_index - 1) // 2

        return (scoped_index, scoped_total)

    def get_effective_gamma(self):
        gamma = float(self.cmhc_gamma)

        if gamma >= 1.:
            return 1.

        if self.cmhc_gamma_scope == "layer":
            return self.get_layer_effective_gamma(gamma)

        if self.hres_index is None or self.num_hres is None:
            return self.get_layer_effective_gamma(gamma)

        scoped_position = self.get_scoped_hres_position(self.cmhc_gamma_scope)
        if scoped_position is None:
            return 1.

        if self.cmhc_gamma_num_layers is None or cmhc_gamma_selects_all(self.cmhc_gamma_num_layers):
            return gamma

        scoped_index, scoped_total = scoped_position
        count = int(self.cmhc_gamma_num_layers)
        count = min(abs(count), scoped_total)

        if count == 0:
            return 1.

        if self.cmhc_gamma_num_layers > 0:
            return gamma if scoped_index < count else 1.

        return gamma if scoped_index >= (scoped_total - count) else 1.

    def get_layer_effective_gamma(self, gamma: float):
        if self.cmhc_gamma_num_layers is None or cmhc_gamma_selects_all(self.cmhc_gamma_num_layers):
            return gamma

        if int(self.cmhc_gamma_num_layers) < 0:
            return gamma

        if self.layer_index is None or self.num_model_layers is None:
            return gamma

        active_start = max(self.num_model_layers - int(self.cmhc_gamma_num_layers), 0)

        if self.layer_index < active_start:
            return 1.

        return gamma

    def set_hres_capture(self, enabled: bool, name: str | None = None):
        _set_hres_capture(self, enabled, name = name)

    def set_hres_stream_capture(self, enabled: bool):
        _set_hres_stream_capture(self, enabled)

    def reset_hres_capture_state(self):
        _reset_hres_capture_state(self)

    def consume_hres_capture_state(self):
        return _consume_hres_capture_state(self)

    def _record_hres_capture_state(
        self,
        alpha_residual: torch.Tensor,
        *,
        permutation_coefficients: torch.Tensor | None = None,
        fixed_identity: bool = False,
    ):
        _record_hres_capture_state(
            self,
            alpha_residual,
            permutation_coefficients = permutation_coefficients,
            coefficient_num_streams = self.num_residual_streams,
            init_permutation_index = self.mhc_init_permutation_indices,
            fixed_identity = fixed_identity,
        )

    def _record_stream_capture_state(self, residuals: torch.Tensor):
        _record_stream_capture_state(self, residuals)

    # Backwards-compatible aliases for the original CMHC-only monitoring hook names.
    def set_monitoring(self, enabled: bool, name: str | None = None):
        self.set_hres_capture(enabled, name = name)

    def reset_monitor_state(self):
        self.reset_hres_capture_state()

    def consume_monitor_state(self):
        return self.consume_hres_capture_state()

    def _record_monitor_state(self, alpha_residual: torch.Tensor):
        self._record_hres_capture_state(alpha_residual)

    def width_connection(
        self,
        residuals
    ):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        # split out fractions

        residuals = self.split_fracs(residuals)

        # split out streams

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = streams)

        # norm
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s = streams)
        # normed = F.normalize(normed, dim = -1)
        normed = self.norm(normed)
        normed_streams = rearrange(normed, 'b ... (s d) -> b ... s d', s = streams)
        self._record_stream_capture_state(normed_streams)

        # alpha for weighted sum of residuals going into branch
        psize = self.num_input_views * streams
        if self.use_identity_hres:
            dynamic_pre = normed @ self.dynamic_alpha_pre_fn
            static_pre = self.static_alpha_pre
            dynamic_residual = None
            static_residual = None
            perms_device = dynamic_pre.device
            perms_dtype = dynamic_pre.dtype
        else:
            wc_weight = normed @ self.dynamic_alpha_fn # ... f (s*v + s!)
            dynamic_pre, dynamic_residual = wc_weight[..., :psize], wc_weight[..., psize:]
            static_pre, static_residual = self.static_alpha[:psize], self.static_alpha[psize:]
            perms_device = wc_weight.device
            perms_dtype = wc_weight.dtype

        perms = get_cached_permutations(
            streams,
            device = perms_device,
            dtype = perms_dtype,
            gamma = self.get_effective_gamma()
        )
        if self.use_identity_hres:
            identity_hres = torch.eye(streams, device = perms_device, dtype = perms_dtype)
            expand_shape = dynamic_pre.shape[:-1] + (streams, streams)
            alpha_residual = identity_hres.view((1,) * (len(expand_shape) - 2) + (streams, streams)).expand(expand_shape)
            res_coeff = None
        else:
            res_coeff = self.residual_scale * dynamic_residual + static_residual
            res_coeff = torch.softmax(res_coeff, dim = -1)
            alpha_residual = einsum(res_coeff, perms, '... r, r i j-> ... i j') # (..., s, s)
        alpha_residual = self.split_fracs(alpha_residual) # (..., f, s, f, s)
        self._record_hres_capture_state(
            alpha_residual,
            permutation_coefficients = res_coeff,
            fixed_identity = self.use_identity_hres,
        )
        
        # (..., f, s, f, v)
        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre  # (..., s*v)
        alpha_pre = rearrange(alpha_pre, '... (f s v) -> ... s f v', v = self.num_input_views, f = self.num_fracs)
        alpha_pre = alpha_pre.sigmoid()

        # the alpha is now split and "manifold constrained" with sinkhorn and sigmoid

        alpha = cat((alpha_pre, alpha_residual), dim = -1) # (..., f, s, f, s+v)

        # beta for weights from branch output back to residual streams

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn # ... (s f)
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s = streams)

            dynamic_beta = dc_weight * self.h_post_scale

            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s = streams)

            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2 # sigmoid * 2 for "H_post"

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        # maybe merge fractions back

        branch_input = self.merge_fracs(branch_input)
        
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        output = rearrange(output, 'b ... s d -> (b s) ... d')

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

CMHCLite.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
CMHCLite.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals
