# MiniCPM-V 4.7 hybrid inference example

This example records the selected INT4 CPU/GPU inference policy with portable path arguments. It uses the SGLang source in this checkout and the Python environment running the launcher. It requires compatible native kernels and `kt-kernel==0.7.2` with the separate multimodal expert-prefix loader fix.

From the repository root, inspect the launch command without starting inference:

```sh
python examples/runtime/minicpmv47/serve.py \
  examples/runtime/minicpmv47/selected.json \
  --work-dir runtime \
  --model-path models/checkpoint \
  --cuda-home toolchain \
  --cache-dir cache \
  --tmp-dir tmp \
  --dry-run
```

These are relative example locations. Supply the actual runtime, model, toolchain, cache, and temporary directories for your environment. `--source-dir` overrides this checkout's Python source; `--python` overrides the executable. The default bind is loopback; `--host`, `--port`, and `--served-model-name` are configurable. Dry-run output redacts model and executable paths. Remove `--dry-run` only after preparing the required runtime and checkpoint.

The profile uses RAWINT4 with `KT_RAWINT4_BACKEND=avxvnni256`, eight CPU threads, 128 resident GPU experts out of 256, zero expert deferral, batch-one decode graphs, 1024-token prefill chunks, prefix caching, and an 8192-token context and token pool. These are model-specific settings, not a general placement policy for every MoE model.

The generic weight-offloader change in this branch belongs to the comparison baseline; this hybrid profile does not enable generic CPU weight offloading. The generic change retains non-expert parameters throughout the isolated runtime and should be scoped or generalized before broader use.

The same launcher and expert placement can serve Qwen3.6-35B-A3B through SGLang's native Qwen3.5 MoE architecture. Use [qwen36.json](qwen36.json), supply an expert-only compressed-tensors INT4 group-128 checkpoint through `--model-path`, and set `--served-model-name qwen3.6-35b-a3b-int4`. The official BF16 checkpoint needs a separate quantization step; changing the launch profile does not quantize it. Local round-to-nearest expert quantization has not established quality equivalence to BF16.

The optional loaded-model audit uses `HYBRID_REFERENCE_AUDIT_PATH`; the launcher also supplies the backward-compatible `MINICPM47_AUDIT_PATH`. Both models report expert placement, CPU implementation, GPU method and runner, conversion state, parameter dtypes, and devices once during the first forward. This inspection does not change routing or disable expert computation. Keep audit files outside the source tree.

CPU routing/loading checks are available in [the manual regression test](../../../test/manual/test_minicpmv47_kt_port.py). GPU validation and measurements must identify the exact source commit, native package versions, checkpoint, launch profile, benchmark client, and token counts. Earlier measurements from another source baseline do not establish this branch's speed or correctness.
