"""CPU regression checks for hybrid loading and routing.

Execute checked-in function bodies with real CPU tensors. Extraction avoids
CUDA-heavy imports; only the transport boundary is stubbed.
"""

import ast
import copy
import json
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterable, Optional, Set, Tuple
from unittest.mock import Mock, patch

import torch

SRT = Path(__file__).resolve().parents[2] / "python/sglang/srt"


def load(path, name, scope, owner=None):
    nodes = ast.parse((SRT / path).read_text()).body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = copy.deepcopy(next(n for n in nodes if getattr(n, "name", None) == name))
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, path, "exec"), scope)
    return scope[name]


def namespace(**kwargs):
    return dict(torch=torch, Optional=Optional, dataclass=dataclass, **kwargs)


class TestMiniCPMHybridPort(unittest.TestCase):
    def test_gpu_mask_preserves_cpu_ids(self):
        mask = load("layers/moe/kt_ep_wrapper.py", "mask_cpu_expert_ids", namespace())
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor([[0, 127, 128, 255]], dtype=dtype)
            before = ids.clone()
            masked = mask(ids, 128)
            self.assertTrue(torch.equal(ids, before))
            self.assertEqual(masked.tolist(), [[0, 127, 0, 0]])
            self.assertNotEqual(masked.data_ptr(), ids.data_ptr())

    def test_async_cpu_and_gpu_contributions_are_preserved(self):
        scope = namespace()
        load("layers/moe/kt_ep_wrapper.py", "mask_cpu_expert_ids", scope)
        apply = load("layers/moe/kt_ep_wrapper.py", "apply", scope, "KTEPWrapperMethod")
        routes = namedtuple("Routes", "topk_ids topk_weights")
        dispatch = namedtuple("Dispatch", "hidden_states topk_output")
        combine = namedtuple("Combine", "hidden_states")
        topk = ModuleType("sglang.srt.layers.moe.topk")
        topk.TopKOutputChecker = SimpleNamespace(format_is_bypassed=lambda _: False)
        transport = ModuleType("sglang.srt.layers.moe.token_dispatcher")
        transport.StandardCombineInput = combine
        for values in ([[0, 1, 2, 3]], [[0, 1, 0, 1]], [[2, 3, 2, 3]]):
            with self.subTest(ids=values):
                ids = torch.tensor(values, dtype=torch.int32)
                weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
                before_ids, before_weights = ids.clone(), weights.clone()
                pending = []

                def submit(_, payload):
                    pending.append(payload.topk_output)

                def gpu(_, payload):
                    routing = payload.topk_output
                    self.assertTrue(bool((routing.topk_ids >= 0).all()))
                    self.assertTrue(bool((routing.topk_ids < 2).all()))
                    return combine(
                        ((routing.topk_ids + 1) * routing.topk_weights).sum(
                            1, keepdim=True
                        )
                    )

                def sync(_):
                    routing = pending[0]
                    return (
                        (routing.topk_ids + 1)
                        * routing.topk_weights
                        * (routing.topk_ids >= 2)
                    ).sum(1, keepdim=True)

                wrapper = SimpleNamespace(
                    tp_rank=0,
                    num_gpu_experts=2,
                    submit=submit,
                    gpu_method=SimpleNamespace(apply=gpu),
                    sync=sync,
                )
                payload = dispatch(torch.ones(1, 1), routes(ids, weights))
                with (
                    patch.dict(
                        sys.modules,
                        {topk.__name__: topk, transport.__name__: transport},
                    ),
                    patch.dict(os.environ, {"MINICPM_KT_AUDIT": "0"}),
                ):
                    result = apply(wrapper, None, payload)
                expected = ((before_ids + 1) * before_weights).sum(1, keepdim=True)
                torch.testing.assert_close(result.hidden_states, expected)
                self.assertTrue(torch.equal(ids, before_ids))
                self.assertTrue(torch.equal(weights, before_weights))

    def test_nested_text_layer_count(self):
        settings = SimpleNamespace(
            kt_weight_path="checkpoint",
            kt_num_gpu_experts=128,
            kt_cpuinfer=8,
            kt_threadpool_count=1,
            kt_method="RAWINT4",
            kt_max_deferred_experts_per_token=0,
        )
        scope = namespace(
            get_exec=lambda: SimpleNamespace(moe=settings),
            get_schedule=lambda: SimpleNamespace(chunked_prefill_size=1024),
            model_config_of=lambda _: SimpleNamespace(
                hf_config=SimpleNamespace(),
                hf_text_config=SimpleNamespace(num_hidden_layers=40),
            ),
        )
        load("layers/moe/kt_ep_wrapper.py", "KTConfig", scope)
        create = load(
            "layers/moe/kt_ep_wrapper.py", "create_kt_config_from_server_args", scope
        )
        self.assertEqual(create(None, 3).num_layers, 40)
        settings.kt_weight_path = None
        self.assertIsNone(create(None, 3))

    def test_missing_metadata_uses_identity_expert_map(self):
        method = load(
            "layers/moe/kt_ep_wrapper.py",
            "process_weights_after_loading",
            namespace(),
            "KTEPWrapperMethod",
        )
        module = ModuleType("sglang.srt.eplb.expert_location_dispatch")
        module.get_global_expert_location_metadata = lambda: None
        loaded = Mock()
        wrapper = SimpleNamespace(
            tp_rank=0,
            global_num_experts=256,
            kt_config=SimpleNamespace(layer_idx=2),
            gpu_method=SimpleNamespace(),
            wrapper=SimpleNamespace(load_weights=loaded),
        )
        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch.object(torch.cuda, "synchronize"),
        ):
            method(wrapper, None)
        torch.testing.assert_close(
            loaded.call_args.args[0], torch.arange(256, dtype=torch.int64)
        )

    def test_gpu_loader_excludes_cpu_experts(self):
        method = load(
            "layers/moe/fused_moe_triton/layer.py",
            "weight_loader",
            namespace(get_global_expert_location_metadata=lambda: None),
            "FusedMoE",
        )
        calls = Mock()
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(
                override_num_local_experts=True, num_gpu_experts=2
            ),
            quant_config=None,
            uses_expert_location=False,
            _map_global_expert_id_to_local_expert_id=lambda value: value,
            _weight_loader_impl=calls,
        )
        parameter = torch.nn.Parameter(torch.zeros(1))
        method(layer, parameter, torch.ones(1), "weight", "w1", 2)
        calls.assert_not_called()
        method(layer, parameter, torch.ones(1), "weight", "w1", 1)
        self.assertEqual(calls.call_args.kwargs["expert_id"], 1)

    def test_minicpm_layout_survives_eval_unwrapping(self):
        tree = ast.parse((SRT / "models/minicpmv.py").read_text())
        wrapper = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniCPMV"
        )
        methods = [
            copy.deepcopy(n)
            for n in wrapper.body
            if getattr(n, "name", None) == "shared_experts_config"
        ]
        cls = ast.ClassDef(
            name="Wrapper", bases=[], keywords=[], body=methods, decorator_list=[]
        )
        scope = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                "minicpmv-layout",
                "exec",
            ),
            scope,
        )
        model = scope["Wrapper"]()
        model.minicpmv = torch.nn.Linear(1, 1)
        layout = object()
        model.shared_experts_config = layout
        self.assertIs(model.shared_experts_config, layout)
        self.assertIs(model.minicpmv.eval().shared_experts_config, layout)

    def test_qwen_packed_metadata_is_not_loaded_as_an_expert(self):
        loader = Mock()
        parameter = SimpleNamespace(weight_loader=loader)
        mapper = SimpleNamespace(apply=lambda weights: weights)
        scope = namespace(
            Iterable=Iterable,
            Tuple=Tuple,
            Set=Set,
            QWEN3_5_KV_SCALE_MAPPER=mapper,
            FusedMoE=SimpleNamespace(
                make_expert_params_mapping=lambda **_: [
                    ("experts.w13_", "experts.0.gate_proj.", 0, "w1")
                ]
            ),
            get_layer_id=lambda _: None,
            default_weight_loader=loader,
            LazyValue=lambda fn: fn,
        )
        method = load(
            "models/qwen3_5.py",
            "load_weights",
            scope,
            "Qwen3_5MoeForConditionalGeneration",
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_experts=1, tie_word_embeddings=False, encoder_only=False
            ),
            enable_shared_expert_fusion=False,
            named_parameters=lambda **_: [
                ("model.layers.0.mlp.experts.w13_weight_packed", parameter)
            ],
        )
        packed = torch.tensor([[1, 2]], dtype=torch.int32)
        entries = [
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_shape",
                torch.tensor([1, 16]),
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed",
                packed,
            ),
        ]
        consumed = []

        def weights():
            for entry in entries:
                consumed.append(entry[0])
                yield entry

        loaded = method(model, weights())
        self.assertEqual(consumed, [name for name, _ in entries])
        self.assertEqual(loaded, {"model.layers.0.mlp.experts.w13_weight_packed"})
        loader.assert_called_once()
        self.assertIs(loader.call_args.args[1], packed)
        self.assertEqual(loader.call_args.kwargs, {"shard_id": "w1", "expert_id": 0})

    def test_runtime_audit_is_optional_and_written_once(self):
        scope = dict(os=os, json=json, Path=Path)
        audit = load("utils/hybrid_runtime_audit.py", "audit_hybrid_runtime", scope)
        method = SimpleNamespace(
            num_gpu_experts=128,
            global_num_experts=256,
            kt_config=SimpleNamespace(
                method="RAWINT4", max_deferred_experts_per_token=0
            ),
            wrapper=SimpleNamespace(moe=object()),
            gpu_method=object(),
        )
        module = SimpleNamespace(
            quant_method=method, scheme=object(), is_marlin_converted=True
        )
        model = SimpleNamespace(
            named_parameters=Mock(return_value=[("weight", torch.ones(2))]),
            named_modules=lambda: [("experts", module)],
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
        ):
            audit(model)
            model.named_parameters.assert_not_called()
            target = Path(directory) / "audit.json"
            os.environ["HYBRID_REFERENCE_AUDIT_PATH"] = str(target)
            audit(model)
            saved = target.read_bytes()
            audit(model)
            self.assertEqual(target.read_bytes(), saved)
            model.named_parameters.assert_called_once()
            result = json.loads(saved)
            self.assertEqual(result["parameters"][0]["shape"], [2])
            layer = result["hybrid_expert_layers"][0]
            self.assertEqual(layer["gpu_experts"], 128)
            self.assertEqual(layer["deferred_experts"], 0)
            self.assertTrue(layer["gpu_marlin_converted"])

    def test_canvas_uses_tokenizer_and_keeps_options(self):
        tokenizer = Mock(return_value=object())
        scope = namespace(
            is_mistral_model=lambda _: False,
            resolve_runai_obj_uri=lambda value: value,
            _normalize_image_processor_backend=lambda *_: "auto",
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(
                    model_type="minicpmv4_7"
                )
            ),
            get_tokenizer=tokenizer,
        )
        method = load("utils/hf_transformers/processor.py", "get_processor", scope)
        method(
            "checkpoint",
            tokenizer_mode="slow",
            tokenizer_revision="revision",
            use_fast=False,
        )
        self.assertEqual(tokenizer.call_args.kwargs["tokenizer_mode"], "slow")
        self.assertEqual(tokenizer.call_args.kwargs["revision"], "revision")
        self.assertEqual(tokenizer.call_args.kwargs["tokenizer_backend"], "huggingface")


if __name__ == "__main__":
    unittest.main()
