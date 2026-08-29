# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns a large, finite vector in
    the input dtype that makes exp(<X, Y>) underflow to zero. Raises a ValueError
    if no such hyperplane is found.

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim), computed in FP32 and
        safely rescaled for the input dtype, such that exp(<X, Y>) ≈ 0.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            # Computing ``-1e5 * Y`` directly in FP16 overflows before the
            # division because FP16's largest finite value is 65504. Perform
            # the calculation in FP32, then rescale the complete vector (rather
            # than clipping individual components) to preserve its direction.
            Y_float = Y.float()
            fake_keys = -1e5 * Y_float / Y_float.norm(dim=-1, keepdim=True).square().clamp_min(1e-12)

            if X.is_floating_point():
                dtype_limit = torch.finfo(X.dtype).max
                safe_limit = min(1e5, dtype_limit / 8)
                largest_component = fake_keys.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
                scale = (safe_limit / largest_component).clamp_max(1.0)
                fake_keys = fake_keys * scale

            return fake_keys.to(dtype=X.dtype)
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k such that exp(<q, k>) = 0 to fake head-wise compression
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        masked_entries = None
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys. `key` is the live
            # cache tensor, so the originals are kept and put back below: left
            # in place, these huge synthetic keys persist in the cache, and the
            # moment a later pass clears masked_key_indices they stop being
            # masked and start participating in attention as if they were real.
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            masked_entries = (batch_indices, head_indices, seq_indices, key[batch_indices, head_indices, seq_indices])
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        try:
            return func(module, query, key, value, attention_mask, dropout, **kwargs)
        finally:
            if masked_entries is not None:
                batch_indices, head_indices, seq_indices, original = masked_entries
                key[batch_indices, head_indices, seq_indices] = original

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
