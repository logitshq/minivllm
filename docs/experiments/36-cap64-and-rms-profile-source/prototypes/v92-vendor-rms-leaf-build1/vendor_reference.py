"""V92 independent checkpoint-gamma reference; import is CPU/stdlib-only.

Runtime prepare is root-only GPU work, before any token. This profile changes
plain RMSNorm leaf rounding only; GDN and GatedRMSNorm retain candidate44 math.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import struct

D = Path(__file__).resolve().parent
CANDIDATE = D.parent / "candidate-v92-vendor-rms-leaf-build1-root"
PROFILE = "vendor RMSNorm leaf rounding"


@dataclass(frozen=True)
class GammaEntry:
    name: str
    norm: object
    source_weight: object
    raw_gamma: object
    cached_gamma: object
    reference: object
    checkpoint_key: str
    raw_sha256: str
    cached_sha256: str
    nbytes: int


def _plain_norms(model):
    if len(model.layers) != 64:
        raise ValueError("V92 requires this 64-layer checkpoint")
    result = []
    for i, layer in enumerate(model.layers):
        for leaf in ("input_layernorm", "post_attention_layernorm"):
            result.append((f"layers.{i}.{leaf}", getattr(layer, leaf)))
        if layer.is_attn:
            for leaf in ("q_norm", "k_norm"):
                result.append((f"layers.{i}.self_attn.{leaf}", getattr(layer.self_attn, leaf)))
    result.append(("norm", model.norm))
    if len(result) != 161 or len({id(n) for _, n in result}) != 161:
        raise ValueError("expected 161 distinct plain RMSNorm objects")
    return result


def _parameter_identity(model, tree_flatten):
    return tuple((name, id(a), tuple(a.shape), str(a.dtype))
                 for name, a in tree_flatten(model.parameters()))


def _small_raw(spec, header_cache):
    path = Path(spec["path"])
    if path not in header_cache:
        with path.open("rb") as f:
            size_raw = f.read(8)
            if len(size_raw) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", size_raw)[0]
            if size > 16 << 20:
                raise ValueError("unexpected header length")
            header = f.read(size)
        if len(header) != size or hashlib.sha256(header).hexdigest() != spec["header_sha256"]:
            raise ValueError("checkpoint header changed")
        header_cache[path] = (size, json.loads(header))
    size, header = header_cache[path]
    entry = header[spec["key"]]
    expected_offsets = [x + 8 + size for x in entry["data_offsets"]]
    if (entry["dtype"] != "BF16" or entry["shape"] != spec["shape"]
            or expected_offsets != [spec["offset"], spec["offset"] + spec["nbytes"]]):
        raise ValueError("checkpoint tensor metadata changed")
    with path.open("rb") as f:
        f.seek(spec["offset"])
        raw = f.read(spec["nbytes"])
    if len(raw) != spec["nbytes"] or hashlib.sha256(raw).hexdigest() != spec["raw_sha256"]:
        raise ValueError("checkpoint gamma range changed")
    return raw


class PreparedProfile:
    def __init__(self, model, entries, norm_class, original_call, parameter_identity,
                 tree_flatten, runtime):
        self.model = model
        self.entries = tuple(entries)
        self._class = norm_class
        self._original_call = original_call
        self._parameter_identity = parameter_identity
        self._tree_flatten = tree_flatten
        self._runtime = runtime
        self._in_reference = False

    def validate_current_bindings(self):
        if _parameter_identity(self.model, self._tree_flatten) != self._parameter_identity:
            raise ValueError("model parameter identities/shapes/dtypes changed")
        for e in self.entries:
            if (e.norm.weight is not e.source_weight
                    or not e.norm._vendor_rms_leaf_current()
                    or object.__getattribute__(e.norm, "_vendor_rms_leaf_gamma") is not e.cached_gamma
                    or e.reference.weight is not e.raw_gamma
                    or e.raw_gamma is e.cached_gamma):
                raise ValueError(f"stale gamma binding: {e.name}")
        return True

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("vendor RMS leaf switch requires strict bool")
        if self._in_reference:
            raise RuntimeError("cannot toggle while reference context is active")
        if enabled:
            self.validate_current_bindings()
        for e in self.entries:
            e.norm.set_vendor_rms_leaf(enabled)

    @contextmanager
    def reference_context(self):
        if self._in_reference or self._class.__call__ is not self._original_call:
            raise RuntimeError("nested/foreign plain RMSNorm patch")
        self.validate_current_bindings()
        entries = {id(e.norm): e for e in self.entries}
        def raw_reference(norm, x):
            e = entries.get(id(norm))
            if e is None or norm.weight is not e.source_weight:
                raise ValueError("unregistered or replaced plain RMSNorm")
            return e.reference(x)
        self._in_reference = True
        self._class.__call__ = raw_reference
        try:
            yield self
        finally:
            self._class.__call__ = self._original_call
            self._in_reference = False

    def owners(self):
        return tuple(record for e in self.entries for record in (
            {"label": f"vendorA.rawgamma.{e.name}", "role": "independent_checkpoint_raw_gamma", "array": e.raw_gamma},
            {"label": f"vendorB.cachedgamma.{e.name}", "role": "offset_derived_nonparameter_gamma", "array": e.cached_gamma}))

    def metadata(self):
        return {"profile": PROFILE, "scope": "plain RMSNorm leaves only; current44 GDN/GatedRMSNorm unchanged",
                "old_gold_pass": False, "whole_omlx_semantics": False,
                "runtime": self._runtime, "prepared_before_any_token_required": True,
                "plain_norms": len(self.entries), "parameter_count": len(self._parameter_identity),
                "new_registered_parameters": 0, "independent_raw_owners": len(self.entries),
                "cached_nonparameter_owners": len(self.entries),
                "gamma_bytes_each_arm": sum(e.nbytes for e in self.entries),
                "enabled_count": sum(object.__getattribute__(e.norm, "_vendor_rms_leaf_enabled") for e in self.entries),
                "entries": [{"name": e.name, "checkpoint_key": e.checkpoint_key,
                             "shape": list(e.raw_gamma.shape), "dtype": str(e.raw_gamma.dtype), "nbytes": e.nbytes,
                             "raw_sha256": e.raw_sha256, "cached_sha256": e.cached_sha256,
                             "raw_is_cached": e.raw_gamma is e.cached_gamma,
                             "norm_id": id(e.norm), "original_weight_id": id(e.source_weight),
                             "raw_gamma_id": id(e.raw_gamma), "cached_gamma_id": id(e.cached_gamma)} for e in self.entries]}

    def snapshot(self):
        return self.metadata()


def prepare_model(model, *, mx, np, nn, device_info, expected_parameter_count=1349):
    """Root GPU prepare after model load, before ANY token. Return profile OFF.

    A gamma is read independently from 161 small checkpoint ranges. B gamma is
    separately derived from live loader offsets, then every byte is verified.
    This function does not enable conv/gates/normreuse; caller retains current44.
    """
    from mlx.utils import tree_flatten
    versions = {n: version(n) for n in ("mlx", "mlx-metal")}
    actual = mx.device_info()
    if (versions != {"mlx": "0.32.3", "mlx-metal": "0.32.3"}
            or mx.default_device() != mx.gpu
            or actual.get("device_name") != "Apple M4 Max"
            or actual.get("architecture") != "applegpu_g16s"
            or device_info.get("device_name") != actual["device_name"]
            or device_info.get("architecture") != actual["architecture"]):
        raise ValueError("V92 runtime/device differs from declared profile")
    norms = _plain_norms(model)
    norm_class = type(norms[0][1])
    if Path(inspect.getfile(norm_class)).resolve() != CANDIDATE / "tutorial_code/miniengine/model/layers.py":
        raise ValueError("model is not the V92 candidate plain RMSNorm class")
    before = _parameter_identity(model, tree_flatten)
    if len(before) != expected_parameter_count or expected_parameter_count != 1349:
        raise ValueError("expected original 1349 model parameters")
    plan = json.loads((D / "raw-gamma-specs.json").read_text())
    specs = {s["key"]: s for s in plan["arrays"]}
    if len(specs) != 161:
        raise ValueError("expected 161 independent checkpoint gamma specs")
    entries = []
    header_cache = {}
    def raw(a):
        mx.eval(a)
        if a.dtype != mx.bfloat16:
            raise ValueError("gamma dtype must be BF16")
        return np.asarray(a.view(mx.uint16)).tobytes()
    for name, norm in norms:
        key = "language_model.model." + name + ".weight"
        spec = specs[key]
        if type(norm) is not norm_class or tuple(norm.weight.shape) != tuple(spec["shape"]):
            raise ValueError(f"plain norm type/shape mismatch: {name}")
        norm.set_vendor_rms_leaf(False)
        source_weight = norm.weight
        blob = _small_raw(spec, header_cache)
        detached = np.frombuffer(blob, dtype="<u2").copy()
        raw_gamma = mx.array(detached, dtype=mx.uint16).view(mx.bfloat16)
        mx.eval(raw_gamma)
        reference = nn.RMSNorm(spec["shape"][0], eps=norm.eps)
        reference.weight = raw_gamma
        cached_gamma = norm.prepare_vendor_rms_leaf(runtime_version=versions["mlx"],
            device_name=actual["device_name"], architecture=actual["architecture"])
        mx.eval(raw_gamma, cached_gamma)
        if (norm.weight is not source_weight or raw_gamma is cached_gamma
                or raw_gamma.shape != cached_gamma.shape or raw_gamma.dtype != cached_gamma.dtype
                or not bool(mx.all(mx.isfinite(raw_gamma)).item())
                or not bool(mx.all(mx.isfinite(cached_gamma)).item())
                or raw(raw_gamma) != blob or raw(cached_gamma) != blob):
            raise ValueError(f"actual raw/derived gamma mismatch: {name}")
        digest = hashlib.sha256(blob).hexdigest()
        entries.append(GammaEntry(name, norm, source_weight, raw_gamma, cached_gamma,
                                  reference, key, digest, digest, len(blob)))
    if _parameter_identity(model, tree_flatten) != before:
        raise ValueError("prepare registered/replaced model parameters")
    profile = PreparedProfile(model, entries, norm_class, norm_class.__call__, before,
                              tree_flatten, {"versions": versions, "device_name": actual["device_name"],
                                             "architecture": actual["architecture"]})
    profile.validate_current_bindings()
    return profile
