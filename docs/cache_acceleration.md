# Cache Acceleration Methods

HunyuanVideo-1.5 supports three cache acceleration methods for skipping redundant DiT
block computation during multi-step denoising inference: **DeepCache**, **TeaCache**, and
**TaylorCache**. All three are implemented in the `angelslim` library (>=0.2.1) and share
the same infrastructure -- monkey-patching each transformer block's `forward` method to
intercept calls and return cached results when possible.

## Common Infrastructure

### Base Class: `CacheHelper`

`angelslim.compressor.diffusion.cache_helper.CacheHelper` is the shared base. It:

1. Wraps every block's `forward` with a `wrapped_forward` that calls `is_skip()` to decide
   whether to reuse cache or recompute.
2. Stores the original forward in `function_dict` and the latest output in `cached_output`.
3. All three concrete helpers inherit from it and override `is_skip` and/or
   `wrap_block_forward`.

### Pipeline Integration

In `hyvideo/pipelines/hunyuan_video_pipeline.py`, the cache helper is instantiated inside
`apply_infer_optimization()` based on `InferState.cache_type`. During denoising the loop
updates `cache_helper.cur_timestep = i` at every step so the helper knows which step it is
on.

### `no_cache_steps` -- Which Steps Force Recomputation

All three methods share the same step-selection logic. A `no_cache_steps` set is computed
once at initialization (pipeline.py:1374):

```python
no_cache_steps = (
    list(range(0, cache_start_step))                                        # warm-up
  + list(range(cache_start_step, cache_end_step, cache_step_interval))      # periodic refresh
  + list(range(cache_end_step, total_steps))                                # tail refinement
)
```

Steps **not** in `no_cache_steps` will attempt to use cached results. The `is_skip()`
method returns `True` (use cache) when the current step is absent from this set.

With default parameters (`start=11, end=45, interval=4, total=50`):

| Step Range | Behavior | Reason |
|---|---|---|
| 0 -- 10 | Force recompute | Early steps have large noise; features change rapidly |
| 11, 15, 19, 23, 27, 31, 35, 39, 43 | Force recompute | Periodic cache refresh to prevent drift |
| 12, 13, 14, 16, 17, 18, ... | Use cache (skip computation) | Features change slowly mid-denoising |
| 45 -- 49 | Force recompute | Final refinement; quality-critical |

Result: ~30 out of 50 steps skip block computation, yielding significant speedup.

---

## Shared CLI Parameters

| Parameter | Default | `InferState` Field | Description |
|---|---|---|---|
| `--enable_cache` | `False` | `enable_cache` | Master switch |
| `--cache_type` | `"deepcache"` | `cache_type` | One of `deepcache`, `teacache`, `taylorcache` |
| `--cache_start_step` | `11` | `cache_start_step` | First step eligible for caching |
| `--cache_end_step` | `45` | `cache_end_step` | Last step eligible for caching |
| `--total_steps` | `50` | `total_steps` | Total denoising steps |
| `--cache_step_interval` | `4` | `cache_step_interval` | Recompute every N steps inside the cache window |

### Mutual Exclusion with Step-Distilled Models

`--enable_step_distill` and `--enable_cache` cannot be used together (generate.py:144):

```python
if args.enable_step_distill and args.enable_cache:
    raise ValueError(
        "Enabling both step distilled model and cache will lead to performance degradation."
    )
```

Step-distilled models already reduce steps to 8-12, so caching provides little benefit
and introduces unnecessary quality loss.

---

## DeepCache

**Paper:** [DeepCache: Accelerating Diffusion Models with Deep Feature Caching (Ma et al., 2024)](https://arxiv.org/abs/2401.09056)

### Principle

The simplest strategy: cache each block's output directly and return it verbatim on skip
steps. No approximation -- just pure replay.

```python
# angelslim/compressor/diffusion/cache/deepcache_helper.py
def wrapped_forward(*args, **kwargs):
    skip = self.is_skip(block_id, blocktype)
    if skip:
        result = self.cached_output[(blocktype, block_id)]   # return cached output
    else:
        result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
        self.cached_output[(blocktype, block_id)] = result    # update cache
    return result
```

### Block-Level Granularity

DeepCache is the only method that supports `no_cache_block_id`, allowing certain blocks to
opt out of caching:

```python
def is_skip(self, block_id, blocktype):
    if self.cur_timestep - self.start_timestep in self.no_cache_steps:
        return False
    if self.no_cache_block_id and blocktype in self.no_cache_block_id:
        if block_id in self.no_cache_block_id[blocktype]:
            return False
    return True
```

The default excludes block 53 (the last double block) because it produces the final output
of the DiT -- skipping it causes the largest quality degradation.

### DeepCache-Exclusive CLI Parameter

| Parameter | Default | Description |
|---|---|---|
| `--no_cache_block_id` | `"53"` | Block IDs to exclude from caching. Supports ranges (`0-5`) or comma-separated lists (`0,1,2,3`). |

### Trade-offs

| Aspect | Evaluation |
|---|---|
| Speedup | Highest among the three |
| Quality loss | Largest (especially without block exclusion) |
| Extra overhead | None |

---

## TeaCache

**Paper:** [TeaCache: Temporal-Aware Caching for Diffusion Models (Cai et al., 2024)](https://arxiv.org/abs/2404.01321)

### Principle

Only the **last block** uses a residual-compensation strategy; all preceding blocks still
do direct replay. The key insight: during denoising the latent changes smoothly, so the
difference (residual) between a block's input and output also changes slowly. TeaCache
stores this residual from the last full-computation step and adds it to the current input
to approximate the current output.

```python
# angelslim/compressor/diffusion/cache/teacache_helper.py
def wrapped_forward(*args, **kwargs):
    skip = self.is_skip()
    if skip:
        if is_last_double_block and not self.single_blocks:
            # Last block: residual compensation
            result = [
                self.cached_input + self.previous_residual,   # input + residual
                self.cached_output[(blocktype, block_id)][1],  # second output replayed
            ]
        else:
            result = self.cached_output[(blocktype, block_id)]  # non-last: direct replay
    else:
        result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
        self.cached_output[(blocktype, block_id)] = result

        # Record residual (only on recompute steps)
        if block_id == 0:
            self.cached_input = kwargs[self.cache_name]
        if is_last_double_block:
            self.previous_residual = result[0] - self.cached_input  # residual = out - in
```

### Why It Works Better Than DeepCache

DeepCache blindly replays the previous output even though the latent has shifted. TeaCache
at least accounts for the shift by applying the residual to the **current** input of the
last block, producing a more faithful approximation while adding negligible cost.

### TeaCache CLI Parameters

TeaCache exposes **no method-exclusive parameters**. Only the shared parameters apply.

### Trade-offs

| Aspect | Evaluation |
|---|---|
| Speedup | Moderate |
| Quality loss | Moderate |
| Extra overhead | One residual tensor (same size as the last block's input) |

---

## TaylorCache

**Paper:** [Accelerating Diffusion Models through Taylor-Expansion-Based Feature Caching (Zhang et al., 2024)](https://arxiv.org/abs/2410.11890)

### Principle

The most accurate approximation. On recompute steps it:

1. **FFT decomposes** the last block's output into low-frequency and high-frequency
   components (cutoff ratio = 0.1).
2. **Computes finite-difference derivatives** up to order 2 for each frequency band.
3. On skip steps, **Taylor-expands** from the last recompute point to predict the current
   output:

```
f(t + delta) ≈ f(t) + f'(t) * delta + f''(t) / 2! * delta²
```

Non-last blocks still use direct replay.

```python
# angelslim/compressor/diffusion/cache/taylorcache_helper.py

# --- On recompute steps ---
self.counter = 0
result = self.function_dict[(blocktype, block_id)](*args, **kwargs)
if is_last_block and self.cur_timestep != self.start_timestep:
    distance = self.cur_timestep - self.last_full_computation_step
    self.taylor_cache.derivatives_computation(
        cached_output, distance=distance,
        low_freqs_order=2, high_freqs_order=2,
    )
    self.last_full_computation_step = self.cur_timestep

# --- On skip steps ---
if is_last_block:
    self.counter += 1
    output = self.taylor_cache.taylor_formula(distance=self.counter)
    result = [output, ...]
```

### Taylor Expansion (per frequency band)

```python
def taylor_formula(self, distance):
    output = 0
    for i in range(num_derivatives):
        coefficient = 1 / math.factorial(i)
        output += coefficient * derivative[i] * (distance ** i)
    return low_freqs_output + high_freqs_output
```

### Derivative Computation

```python
def derivatives_computation(self, x, distance, low_freqs_order, high_freqs_order):
    x_low, x_high = decomposition_FFT(x, cutoff_ratio=0.1)  # FFT split
    self.set_temp_derivative(0, "low_freqs", x_low)           # 0th order = value
    for i in range(low_freqs_order):
        diff = temp_derivative[i] - old_derivative[i]
        self.set_temp_derivative(i + 1, "low_freqs", diff / distance)  # finite difference
    # same for high_freqs
```

### Internal (Non-CLI) Parameters

These are hardcoded in `TaylorCacheHelper.__init__` and **not exposed as CLI arguments**:

| Parameter | Default | Description |
|---|---|---|
| `max_order` | `2` | Maximum Taylor expansion order |
| `low_freqs_order` | `2` | Derivative order for low-frequency band |
| `high_freqs_order` | `2` | Derivative order for high-frequency band |
| `cutoff_ratio` | `0.1` | FFT frequency boundary (in `decomposition_FFT`) |

To change these, either modify `angelslim` source or construct `TaylorCacheHelper` directly.

### Why Frequency Decomposition Helps

Low-frequency components (spatial structure) change slowly and are well-approximated by low-
order polynomials. High-frequency components (fine detail) change faster but can still be
tracked by their own set of derivatives. Separating them yields a better Taylor fit than
applying a single expansion to the whole tensor.

### TaylorCache CLI Parameters

TaylorCache exposes **no method-exclusive CLI parameters**. Only the shared parameters apply.

### Trade-offs

| Aspect | Evaluation |
|---|---|
| Speedup | Lowest among the three (FFT + derivative computation on recompute steps) |
| Quality loss | Smallest |
| Extra overhead | FFT decomposition + per-band derivative storage (2x2 orders) |

---

## Comparison Summary

| Method | Skip-Step Approximation | Quality | Speedup | Extra Cost | Exclusive Params |
|---|---|---|---|---|---|
| **DeepCache** | Direct output replay | Lowest | Highest | None | `--no_cache_block_id` |
| **TeaCache** | Last block: `input + residual`; others: replay | Medium | Medium | One residual tensor | None |
| **TaylorCache** | Last block: per-band Taylor expansion; others: replay | Highest | Lowest | FFT + derivative tensors | None (internal only) |

All three share the same pattern: non-last blocks are always direct replay. Only the last
block applies the method-specific approximation, because its output most directly affects
the denoising result quality.

## Usage Examples

```bash
# DeepCache with default settings
torchrun --nproc_per_node=8 generate.py \
  --prompt "A cat playing" \
  --resolution 480p \
  --enable_cache --cache_type deepcache

# DeepCache excluding more blocks from caching
torchrun --nproc_per_node=8 generate.py \
  --prompt "A cat playing" \
  --resolution 480p \
  --enable_cache --cache_type deepcache \
  --no_cache_block_id 50-53

# TeaCache (higher quality, same speedup config)
torchrun --nproc_per_node=8 generate.py \
  --prompt "A cat playing" \
  --resolution 480p \
  --enable_cache --cache_type teacache \
  --cache_step_interval 3

# TaylorCache (best quality, conservative interval)
torchrun --nproc_per_node=8 generate.py \
  --prompt "A cat playing" \
  --resolution 480p \
  --enable_cache --cache_type taylorcache \
  --cache_start_step 15 --cache_end_step 42
```
