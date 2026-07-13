"""Unit tests for utils/config.py -- no GPU, no model."""
import sys, os, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from utils.config import GlobalConfig


def write_yaml(data):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Schema round-trips: write a value, load it, assert you get it back.
# These tests are independent of what any checked-in yaml currently says.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["default", "megakernel_cuda"])
def test_model_backend_round_trips(backend):
    tmp = write_yaml({"model": {"backend": backend}})
    try:
        cfg = GlobalConfig.from_yaml(tmp)
        assert cfg.model.backend == backend
    finally:
        os.unlink(tmp)


@pytest.mark.parametrize("flag", [True, False])
def test_use_cuda_graph_round_trips(flag):
    tmp = write_yaml({"model": {"use_cuda_graph": flag}})
    try:
        assert GlobalConfig.from_yaml(tmp).model.use_cuda_graph == flag
    finally:
        os.unlink(tmp)


@pytest.mark.parametrize("kv_backend", ["default", "kivi"])
def test_kv_cache_backend_round_trips(kv_backend):
    tmp = write_yaml({"model": {"kv_cache": {"backend": kv_backend}}})
    try:
        assert GlobalConfig.from_yaml(tmp).model.kv_cache.backend == kv_backend
    finally:
        os.unlink(tmp)


def test_kv_cache_kivi_fields_round_trip():
    data = {"model": {"kv_cache": {
        "backend": "kivi", "k_bits": 4, "v_bits": 4,
        "group_size": 64, "residual_length": 16,
    }}}
    tmp = write_yaml(data)
    try:
        cfg = GlobalConfig.from_yaml(tmp)
        assert cfg.model.kv_cache.k_bits == 4
        assert cfg.model.kv_cache.group_size == 64
        assert cfg.model.kv_cache.residual_length == 16
    finally:
        os.unlink(tmp)


@pytest.mark.parametrize("policy", ["fifo", "spf", "ljf", "random"])
def test_admission_policy_round_trips(policy):
    tmp = write_yaml({"batch": {"admission_policy": policy}})
    try:
        assert GlobalConfig.from_yaml(tmp).batch.admission_policy == policy
    finally:
        os.unlink(tmp)


def test_chunked_prefill_none_by_default():
    tmp = write_yaml({})
    try:
        assert GlobalConfig.from_yaml(tmp).batch.max_num_batched_tokens is None
    finally:
        os.unlink(tmp)


def test_chunked_prefill_set():
    tmp = write_yaml({"batch": {"max_num_batched_tokens": 512}})
    try:
        assert GlobalConfig.from_yaml(tmp).batch.max_num_batched_tokens == 512
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# _base inheritance: child overrides parent, non-overridden fields inherited
# ---------------------------------------------------------------------------

def test_base_inheritance():
    base = write_yaml({"model": {"backend": "default"}, "generation": {"max_new_tokens": 64}})
    child_data = {"_base": base, "generation": {"max_new_tokens": 128}}
    child = write_yaml(child_data)
    try:
        cfg = GlobalConfig.from_yaml(child)
        assert cfg.generation.max_new_tokens == 128   # child overrides
        assert cfg.model.backend == "default"          # parent inherited
    finally:
        os.unlink(base)
        os.unlink(child)


# ---------------------------------------------------------------------------
# Legacy `inference:` migration still produces correct config
# ---------------------------------------------------------------------------

def test_legacy_inference_migration():
    data = {
        "env": {"device": "cpu", "default_dtype": "float32"},
        "inference": {
            "backend": "default",
            "max_new_tokens": 64,
            "stop_on_eos": False,
            "use_cuda_graph": True,
            "kv_cache": {"backend": "default", "max_len": 2048},
        },
    }
    tmp = write_yaml(data)
    try:
        cfg = GlobalConfig.from_yaml(tmp)
        assert cfg.model.backend == "default"
        assert cfg.generation.max_new_tokens == 64
        assert cfg.generation.stop_on_eos is False
        assert cfg.model.use_cuda_graph is True
        assert cfg.model.kv_cache.max_len == 2048
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# run configs: only verify structural invariants, not current default values
# ---------------------------------------------------------------------------

def test_run_configs_load_without_error():
    """All checked-in run configs must be parseable."""
    for name in ("default.yaml", "batch.yaml"):
        path = os.path.join("configs", "runs", name)
        cfg = GlobalConfig.from_yaml(path)
        assert cfg.path.model_path        # non-empty
        assert cfg.model.kv_cache.max_len > 0
        assert cfg.batch.num_slots > 0


def test_model_override_switches_model_path():
    """--model qwen3_5 must produce a different path than the default."""
    cfg_default = GlobalConfig.from_yaml("configs/runs/batch.yaml")
    cfg_override = GlobalConfig.from_yaml("configs/runs/batch.yaml", model="qwen3_5")
    assert cfg_default.path.model_path != cfg_override.path.model_path
