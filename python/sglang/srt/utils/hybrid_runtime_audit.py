"""Opt-in, one-shot inspection of loaded hybrid expert modules."""

import json
import os
from pathlib import Path


def audit_hybrid_runtime(model):
    target = os.environ.get("HYBRID_REFERENCE_AUDIT_PATH") or os.environ.get(
        "MINICPM47_AUDIT_PATH"
    )
    if not target or getattr(model, "_hybrid_runtime_audited", False):
        return
    parameters = [
        {
            "name": name,
            "dtype": str(param.dtype),
            "device": str(param.device),
            "shape": list(param.shape),
            "numel": param.numel(),
            "bytes": param.numel() * param.element_size(),
        }
        for name, param in model.named_parameters()
    ]
    experts = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None and hasattr(method, "kt_config"):
            experts.append(
                {
                    "name": name,
                    "gpu_experts": method.num_gpu_experts,
                    "total_experts": method.global_num_experts,
                    "cpu_method": method.kt_config.method,
                    "cpu_backend": type(method.wrapper.moe).__name__,
                    "gpu_method": type(method.gpu_method).__name__,
                    "gpu_scheme": type(
                        getattr(module, "scheme", None) or method.gpu_method
                    ).__name__,
                    "gpu_runner_backend": str(
                        getattr(
                            getattr(method.gpu_method, "runner", None),
                            "runner_backend",
                            None,
                        )
                    ),
                    "gpu_marlin_converted": bool(
                        getattr(module, "is_marlin_converted", False)
                    ),
                    "deferred_experts": method.kt_config.max_deferred_experts_per_token,
                }
            )
    Path(target).write_text(
        json.dumps(
            {"parameters": parameters, "hybrid_expert_layers": experts}, indent=2
        )
        + "\n"
    )
    model._hybrid_runtime_audited = True
