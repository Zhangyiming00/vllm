"""Capture KV budgeting against vLLM's actual normalized cache groups."""
import copy
import math
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups, get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs


def capture_kv_budget(config, specs):
    if not specs:
        return 1, {'groups': [], 'required_blocks': 1}
    extra = config.additional_config or {}
    batch = int(extra.get('sae_capture_batch_size') or config.scheduler_config.max_num_seqs)
    context = int(extra.get('sae_capture_context_size') or config.model_config.max_model_len)
    if batch <= 0 or context <= 0 or context > config.model_config.max_model_len:
        raise ValueError('Invalid activation capture batch/context')
    # Explicit token capacity is a lower bound, never permission to undersize
    # the actual capture batch. Round per sequence, not after summing tokens.
    explicit = extra.get('sae_capture_kv_pool_capacity_tokens')
    if explicit is not None:
        if int(explicit) <= 0:
            raise ValueError('Capture KV capacity must be positive')
        batch = max(batch, math.ceil(int(explicit) / context))
    groups = get_kv_cache_groups(config, copy.deepcopy(specs))
    # Use configured max_model_len, including the generated token. Mamba's
    # own max_memory_usage_bytes knows its state retention/cache mode.
    if len(groups) == 1 and isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        spec = groups[0].kv_cache_spec
        page = spec.page_size_bytes
        per_request = max(math.ceil(s.max_memory_usage_bytes(config) / s.page_size_bytes)
                          for s in spec.kv_cache_specs.values())
        pool_bytes = page
    else:
        page = groups[0].kv_cache_spec.page_size_bytes
        assert all(g.kv_cache_spec.page_size_bytes == page for g in groups)
        per_request = sum(math.ceil(g.kv_cache_spec.max_memory_usage_bytes(config) / page)
                          for g in groups)
        pool_bytes = page * max(len(g.layer_names) for g in groups)
    required = batch * per_request + 1  # shared BlockPool reserves a null block
    budget = required * pool_bytes
    allocated = get_kv_cache_config_from_groups(config, groups, budget)
    if allocated.num_blocks < required:
        raise ValueError(f'Capture KV override provides {allocated.num_blocks} blocks; need {required}')
    # Overrides increasing the block count must also be included in the budget.
    budget = max(budget, sum(t.size for t in allocated.kv_cache_tensors))
    return budget, dict(batch=batch, max_model_len=config.model_config.max_model_len,
        blocks_per_request=per_request, required_blocks=required,
        allocated_blocks=allocated.num_blocks, bytes=budget,
        groups=[dict(type=type(g.kv_cache_spec).__name__, layers=len(g.layer_names),
                     block_size=g.kv_cache_spec.block_size,
                     page_bytes=g.kv_cache_spec.page_size_bytes) for g in groups])
