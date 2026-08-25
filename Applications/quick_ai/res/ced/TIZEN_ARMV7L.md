# CED on Tizen armv7l

Running the CED audio detector on a 32-bit Tizen userspace over a Cortex-A76.
This is the record of what had to change, what the numbers were, and what is
still on the table.

## Target

The device reports `armv7l` and runs a 32-bit userspace, but the core is a
Cortex-A76. Its `/proc/cpuinfo` Features line is

    half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32
    lpae evtstrm aes pmull sha1 sha2 crc32

Note what is *not* there: no `asimddp`. FEAT_DotProd is not reachable from
AArch32 on this part, so VSDOT is unavailable even though the core implements
it in AArch64. An `-march=armv8.2-a+dotprod` build dies with SIGILL. See
[Dot product](#dot-product).

Build:

    gbs build -A armv7l \
      --define "arm_tune -march=armv8-a+crc -mtune=cortex-a76 -mfpu=neon-fp-armv8 -mfp16-format=ieee" \
      --define "_without_blas 1" \
      --define "_without_opencv 1" \
      --define "_without_tflite 1"

`arm_tune` is appended to CFLAGS/CXXFLAGS last in `%build`, so it beats both
the Tizen reference optflags (which target Cortex-A8) and the `-march` that
meson.build hardcodes for aarch64.

Sanity check on the result:

    objdump -d /usr/lib/libnntrainer.so | grep -c vsdot     # must be 0

## What was broken

Three defects stopped it running at all, none of them in the quantization
path.

### copy_fp32_u16 was a numeric cast on ARM

`arm_compute_backend.cpp` routed it to `__fallback_copy_fp32_u16`, which is
`static_cast<uint16_t>(X[i])`. x86 routes the same call through
`avx2::copy_f32_f16`: a uint16_t destination for a float source means an fp16
*bit pattern*, not a numeric conversion. Every value below 1.0 became 0 and
every negative one wrapped.

`mha_core`'s KV cache is UINT16 whenever `ENABLE_FP16` is off, and on the
`!use_rope` path -- which is what a ViT takes -- it is filled by exactly this
copy. So attention read back garbage: layer 0's output was `{0, NaN}` and the
12 class scores came out nearly constant across windows. AArch64 never hit it,
having `_FP16` and therefore an FP16-typed cache that takes a different path.

### Four attention kernels were NYI in the fallback backend

`compute_kcaches`, `compute_fp16vcache_fp32_transposed`, `softmax_row` and
`softmax_row_inplace` all threw `std::runtime_error("NYI")`.
`arm_compute_backend` routes to them when the target is neither aarch64 nor
`ENABLE_FP16`, which is exactly this build. Implemented against the AVX2
versions and verified equal on the host (max|diff| 2e-7 and 0).

One trap: `local_window_size` defaults to `UINT_MAX`, so the window tests have
to stay in the unsigned domain. Narrowing to `int` first turns the sentinel
into -1 and inverts every comparison -- which is what the first version did,
producing an all-zero vcache output.

### The tokenizer archive is x86_64 only

`Applications/quick_ai/lib` ships one prebuilt `libtokenizers_c.a` and it is
x86_64, while `Applications/meson.build` enters `quick_ai` unconditionally.
`-Dquick_ai-tokenizer=false` (spec: on by default off for ARM) swaps in
throwing factories. Models with `skip_tokenizer` -- CED and the other audio
classifiers -- never construct one.

## Where the time actually goes

`NNTR_PROFILE_LAYERS=1` accumulates wall time per layer type and prints a
sorted table at exit. Use it before optimising anything: the FLOP arithmetic
said the GEMMs should be 94% of inference and they were 42%.

The first profile (one window, qemu-arm) read:

    fully_connected        227.7 ms   42.4%   73 calls
    activation             117.4 ms   21.9%   12 calls    <- scalar erf
    mha_core                71.1 ms   13.2%   12 calls
    layer_normalization     59.2 ms   11.0%   26 calls
    conv2d                  26.4 ms    4.9%    1 call
    xi_pooling              25.5 ms    4.7%    1 call

`activation` being second is what a missing `#else` looks like:

    void gelu_v2(const unsigned int N, const float *X, float *Y) {
    #ifdef __ARM_NEON
      nntrainer::neon::gelu_v2(N, X, Y);
    #endif
      __fallback_gelu_v2(N, X, Y);      // overwrote everything above
    }

The NEON kernel ran and the scalar `std::erf` loop then overwrote every value
it had just written. A sweep of `arm_compute_backend.cpp` found `gelu_v2` to
be the only function with that shape.

## The int8 GEMM

`gemmPerChannelA8` called its dot helper once per 32-value Q8_0 block, so
every block built a vector accumulator and folded it straight back to a
scalar -- 24 horizontal reductions per output element for the 768-wide MLP
instead of one. It also had no row blocking, so with 72 tokens each weight row
was read 72 times.

Keeping the accumulators live across the whole reduction and tiling rows
against one weight load, measured as guest instructions for one CED GEMM
(M=72, N=64) under qemu-arm:

                     K=192      K=768
    before          687466    2595178
    4 rows          505256    1770152     1.36x / 1.47x
    8 rows          477614    1690670     1.44x / 1.53x
    4 rows x 2 cols 539130    1990650     1.28x / 1.30x

With `+dotprod` the ranking inverts -- 4 rows is 2.00x/2.19x and 8 rows only
1.54x/1.83x -- because that path also holds two activation vectors per row and
eight accumulators do not fit AArch32's 16 Q registers. So the tile height
follows `__ARM_FEATURE_DOTPROD`.

Without dotprod, **four int8 MACs per instruction is the ceiling**:
`vmull_s8` produces eight int16 products and `vpadalq_s16` folds them into the
int32 accumulator, two instructions per eight MACs. VSDOT would be sixteen per
instruction. The kernel currently reaches 2.09 MACs/instruction, so roughly
half the instruction stream is loads and loop control.

## Front end

Two problems, both structural.

**Memory grew with the clip.** Every window's mel spectrogram was computed
before any was inferred, and `readWav16` decoded the whole file, so the
resident set rose about 0.15 MB per second of audio. `inferWindows` was
already running one `incremental_inference` per window, so the batching bought
nothing. Windows now stream and `WavStream` pulls each one off disk: 3 s and
27 s clips both sit at 14.1 MB anonymous where 27 s used to cost 17.4 MB.

**The mel filterbank is 3% non-zero** -- 501 of 16448 entries for 64 mels over
257 bins -- and was applied as a dense matrix walked with an `n_mels` stride,
so every load missed. Storing each filter as its contiguous run of non-zero
weights, plus NEON on the windowing and the power spectrum, took the front end
from 2.901 to 1.017 ms/window on x86.

## Numbers

Device, `16k_0128_barking_only.wav`, 13 windows:

| | latency | notes |
|---|---|---|
| before this work | 3.9 s | 4 threads |
| after | 2.78 s | 2 threads |

Thread scaling on the device is not monotonic:

    1 thread  4.6 s
    2 threads 3.6 s
    4 threads 3.9 s

Four is slower than two. Every `parallel_for` is a barrier and waits for the
slowest chunk, so this is what unequal cores look like; it has not been
confirmed against the actual core layout. **Run with 2 threads** until it is.

Memory on the device: 20300 KB. The TFLite C runtime this model came from uses
about 12.9 MB PSS, so there is still a gap. From the host breakdown the pools
themselves are small -- weights 6.034 MB (exactly the file size, so nothing is
dequantized or duplicated at load) and activations 1.44 MB with the v1
planner, 1.28 MB with v3 -- and most of the rest is library text, of which
`libquick_ai.so` is ~3 MB for a model whose own code is 77 KB.

Accuracy is unchanged throughout. Against the reference TFLite runtime, all
four clips, 67 windows: detections 100% identical, top-1 65-66/67, max
per-class delta 0.024-0.031 against a 0.05 budget.

YOLOv7-tiny shares the int8 kernel and was re-run on this branch to confirm it
did not regress. Against the expected output recorded with the model package:

    field   expected   here
    x1         226.3   226.864
    y1           2.0     1.896
    x2         317.9   318.129
    y2         108.6   108.205
    conf       0.057    0.0564
    cls            2         2

The remaining difference is summation order: the recorded baseline came from a
`+dotprod` build whose VSDOT kernel accumulates in a different order than the
vmull/vpadal one, and small float differences amplify through the network. The
tiled kernel itself is element-for-element identical to what it replaced (0
mismatches out of 4608 outputs, K=192 and K=768, on x86 and on armv7l under
both `-march` settings).

## Decisions and why

### Dot product

Not used. `-march=armv8.2-a+dotprod` builds and passes under qemu but SIGILLs
on the device, which reports no `asimddp`. Runtime dispatch on
`getauxval(AT_HWCAP2)` would be the answer on a part that does expose it; here
there is nothing to dispatch to.

### OpenBLAS

**CED does not need it. YOLOv7 does. They cannot currently share one rpm.**

For CED, measured with `OMP_NUM_THREADS=1` to keep OpenMP's spin-wait out of
the comparison, BLAS affects exactly one layer:

    layer                BLAS on   BLAS off
    fully_connected       332.66     331.04
    mha_core              128.08     128.08
    layer_normalization    56.90      56.92
    conv2d                 18.18      24.37

6.2 ms out of ~630. Every FC in this model is Q8_0 and goes through the int8
kernel; only the FP32 patch-embedding conv reaches sgemm. That 1% is not worth
~2 MB resident, so the CED build turns it off.

#### Why YOLOv7 still needs it

Four convolutions in YOLOv7-tiny never become int8 and stay on FP32 sgemm:

- **the stem**, which has 3 input channels. conv2d has a dedicated direct FP32
  path for `in_ch == 3` precisely because a 27-element im2col row is not worth
  the col-buffer machinery, but the fallback for anything it does not catch is
  sgemm.
- **the three detection heads**, which have `out_ch = 27` (3 anchors x (4 box
  + 1 obj + 4 class)). The per-channel W8A8 conv requires `out_ch % 32 == 0`
  because the Q8_0 block is 32 values wide, so 27 is structurally ineligible --
  no amount of tuning makes those quantize.

With BLAS off those four hit `__fallback_sgemm`, which is a naive triple loop
accumulating in `double` with the `TransA`/`TransB` test *inside* the innermost
loop. That is what took an earlier armv7l YOLOv7 build to 9 s per inference.

#### To drop OpenBLAS for YOLOv7, fix this

`__fallback_sgemm` in `nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp`:

    for m, for n:
      double c = 0.0;                        // double accumulate
      for k:
        a = TransA ? A[k*lda+m] : A[m*lda+k];  // branch in the inner loop
        b = TransB ? B[n*ldb+k] : B[k*ldb+n];
        c += a * b;

Three things are wrong with it as a fallback rather than a reference: the
`double` accumulator (VFP double on A32 is far slower than float and blocks
NEON), the transpose test in the innermost loop, and no blocking or
vectorization at all. Hoisting the four transpose cases into separate loops
and giving the `!TransA && !TransB` one a NEON kernel with an accumulator held
across `k` -- the same shape as `dotI8Tile` in `int8_gemm.h` -- would make BLAS
unnecessary for both models and save the ~2 MB.

Until then, build the rpm to match the model:

    CED     --define "_without_blas 1"      # 1% slower, ~2 MB smaller
    YOLOv7  (leave BLAS on)

### Do not trust qemu for anything involving threads

It misled this work twice. A BLAS-on profile showed `layer_normalization` at
809 ms against 57 ms with BLAS off -- entirely OpenMP spin-wait under
emulation, and identical once `OMP_NUM_THREADS=1` controlled for it. Guest
instruction counts (`-singlestep -d exec,nochain`) are reliable and are what
the kernel numbers above come from; wall clock and anything crossing a thread
barrier is not.

## Still open

- **layer_normalization walks the tensor nine times** (average, subtract,
  pow(2), average, add, pow(-0.5), multiply, multiply, add), each a separate
  dispatch and a separate pass over memory. Fusing to one pass is
  architecture-independent and is the most promising remaining item.
- **`xi_pooling` costs 25 ms in a single call** and has not been looked at.
- **`mha_core`'s wrappers call `parallel_for(0, num_cache_head)`**, and CED has
  three heads. Splitting three items across two threads and paying a barrier
  for it is probably not worth it; `NNTR_PARALLEL_MIN` can be swept to find
  out.
- **Why 4 threads is slower than 2.** Check whether the cores are
  heterogeneous (`CPU part` per processor in `/proc/cpuinfo`).
- **The memory gap to TFLite.** `libquick_ai.so` links the whole CausalLM
  stack -- tokenizer 492 KB, minja 382 KB, nlohmann 372 KB -- for an audio
  classifier that uses none of it. A lean target would recover most of 3 MB.

## Tools

| | |
|---|---|
| `NNTR_PROFILE_LAYERS=1` | per-layer-type wall time at exit |
| `NNTR_DUMP_LAYERS=1` | per-layer output fingerprint (mean/rms/min/max) |
| `NNTR_POOL_REPORT=1` | weight and activation pool sizes |
| `NNTR_MEMORY_PLANNER` | `v1` (default), `v2`, `v3`, `basic` |
| `NNTR_NUM_THREADS` | worker threads; 2 on this device |
| `NNTR_PARALLEL_MIN` | below this iteration count, `parallel_for` runs serially |

`NNTR_DUMP_LAYERS` is what located the attention NaN: run it on a working
target and on the port, and diff the first layer whose numbers move.
