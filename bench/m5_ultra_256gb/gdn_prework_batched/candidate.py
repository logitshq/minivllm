"""Default-OFF B1–8 GDN prework adapter, independent of attention/weight packing.

Import is stdlib only. `make_decode` is dependency-injected for CPU contracts;
`install` verifies the real native sources before lazily loading existing fused
kernels. No new kernel math, request cache, evaluation, or global monkeypatch.
Install on a fresh idle model; restore after requests/observers have drained.
Do not retain compiled graphs built inside the scope or combine with E024 yet.
This batch extension is source-prepared only; no numerical or speed qualification.
"""
import hashlib
import importlib
from pathlib import Path
from types import FunctionType, MethodType

PINS = {
    'model/gdn.py': 'c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030',
    'model/layers.py': 'c2c9936df2a34877edb5e1702152583e54549f69b37d7c93e9996af1e7c20908',
    'model/qwen3_5.py': 'dae9694d717a0ebe80c9c54317c65f5d206e4d5695bc898fe0e93eb04a8590ce',
    'cache.py': '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791',
    'context.py': '134614bd19cedae748b0db415cf1feac70cebd9826a641a843936db8327d6ba7',
    'kernels/gdn_metal.py': '51757224b18a424eaee5ebb7346075837bf768c289b8721f6279b6d844de13a9',
    'kernels/gdn_decode_fused.py': '16389d6dd99103db2a4d6649c5c2ab45ce94c966a24d9643ed9ad16bcb77a500',
    'kernels/gdn_norm_reuse_metal.py': 'b42aaceb140c53d87b9d5c5ebc6392a1e1ca0bcf7b55266afc6f68b22e0defd1',
}
QUANT = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}


def _need(ok, message):
    if not ok: raise ValueError(message)


def _load_fused(module):
    path = Path(module.__file__).resolve(); root = path.parent.parent
    _need(path == root/'model/gdn.py', 'actual native GDN source required')
    for relative, digest in PINS.items():
        _need(hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, 'native source changed: '+relative)
    cls = module.GatedDeltaNet
    for name in ('__call__', '_decode', '_gates', '_decode_gates'):
        fn = getattr(cls, name)
        _need(type(fn) is FunctionType and fn.__closure__ is None and fn.__globals__ is vars(module)
              and Path(fn.__code__.co_filename).resolve() == path, 'unrecognized loaded native function: '+name)
    package = module.__package__.rsplit('.', 1)[0]
    fused_module = importlib.import_module(package+'.kernels.gdn_decode_fused')
    _need(Path(fused_module.__file__).resolve() == root/'kernels/gdn_decode_fused.py'
          and fused_module.mx is module.mx, 'mixed fused runtime/source')
    nr = fused_module.nr
    _need(Path(nr.__file__).resolve() == root/'kernels/gdn_norm_reuse_metal.py', 'mixed state kernel source')
    for fn, owner, file in ((fused_module.fused, fused_module, 'gdn_decode_fused.py'),
                           (nr._state, nr, 'gdn_norm_reuse_metal.py')):
        _need(type(fn) is FunctionType and fn.__globals__ is vars(owner)
              and Path(fn.__code__.co_filename).resolve() == root/'kernels'/file,
              'unrecognized existing fused callable')
    return fused_module.fused


def make_decode(module, fused, *, enabled=False):
    """Return an unbound `_decode` function; does not install or import MLX.

    Only the real `install` entry source-verifies modules. The CPU factory is
    explicitly injectable, not a substitute for real-source/device qualification.
    `calls` counts Python calls, never claims execution/completion inside graphs.
    """
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    original = module.GatedDeltaNet._decode
    counts = {'candidate': 0, 'fallback': 0}

    def decode(self, x, st, *, _compile_gates=False, _norm_reuse_normal=False):
        mx = module.mx
        B = x.shape[0] if enabled and len(x.shape) == 2 else 0
        eligible = (enabled and _norm_reuse_normal is True and _compile_gates is True
                    and type(self) is module.GatedDeltaNet
                    and object.__getattribute__(self, '_norm_reuse_enabled') is False
                    and type(st) is module.BatchState and set(vars(st)) == {'conv', 'S'}
                    and (self.Hk, self.Hv, self.Dk, self.Dv, self.conv_dim) == (16,48,128,128,10240)
                    and 1 <= B <= 8 and x.shape == (B,5120) and x.dtype == mx.bfloat16
                    and st.conv.shape == (B,3,10240) and st.conv.dtype == mx.bfloat16
                    and st.S.shape == (B,48,128,128) and st.S.dtype == mx.float32
                    and self.conv1d_weight.shape == (10240,4) and self.conv1d_weight.dtype == mx.bfloat16
                    and self.A_log.shape == self.dt_bias.shape == (48,)
                    and self.A_log.dtype == self.dt_bias.dtype == mx.float32
                    and mx.default_device() == mx.gpu and module._m5_ultra_decode_identity())
        if eligible:
            for name, width, height in (('in_proj_qkv',5120,10240),('in_proj_z',5120,6144),
                    ('in_proj_b',5120,48),('in_proj_a',5120,48),('out_proj',6144,5120)):
                op = getattr(self, name)
                if not (type(op) is module.Linear and op.q == QUANT and op.in_features == width
                        and op.out_features == height and getattr(op,'biases',None) is None
                        and op.weight.shape == (height,width//8) and op.weight.dtype == mx.uint32
                        and op.scales.shape == (height,width//16) and op.scales.dtype == mx.uint8):
                    eligible = False; break
        if not eligible:
            counts['fallback'] += 1
            return original(self,x,st,_compile_gates=_compile_gates,_norm_reuse_normal=_norm_reuse_normal)
        # Preserve all five original Linear calls and the original gate branch.
        # No derived packs/parameter cache: every call reads current attributes.
        # The original _decode_gates itself falls back to _gates for B > 1.
        raw = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B,self.Hv,self.Dv)
        w = self.conv1d_weight.astype(raw.dtype)
        if object.__getattribute__(self,'_gates_compile_enabled'):
            beta,g = self._decode_gates(self.in_proj_b(x),self.in_proj_a(x),x,st)
        else:
            beta,g = self._gates(self.in_proj_b(x),self.in_proj_a(x))
        o,state,conv = fused(raw,st.conv,w,g,beta,st.S)
        # As in native _decode: commit both outputs before norm/out projection.
        # Never catch-and-retry against this advanced state after an exception.
        st.conv,st.S = conv,state
        counts['candidate'] += 1
        return self.out_proj(self.norm(o.astype(x.dtype),z).reshape(B,-1))

    decode.calls = counts
    return decode


class Installation:
    def __init__(self, layer, function):
        self.layer = layer; self.wrapper = MethodType(function,layer)
        self.calls = function.calls; self.closed = False
        object.__setattr__(layer,'_decode',self.wrapper)

    def restore(self):
        if self.closed: return
        _need(vars(self.layer).get('_decode') is self.wrapper, 'restore foreign observer before adapter')
        object.__delattr__(self.layer,'_decode')
        self.layer = self.wrapper = None; self.closed = True

    def __enter__(self):
        _need(not self.closed,'scope already closed'); return self

    def __exit__(self, *exc):
        self.restore(); return False


def install(layer, module, *, enabled=False):
    """OFF returns None untouched; ON temporarily replaces only this `_decode`.

    Caller must install before requests/observers/graphs, target-only, and drain
    or tear down the model before restoration. No generic tracing detector or
    safe live parameter hot-reload protocol is claimed.
    """
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    _need(type(layer) is module.GatedDeltaNet and '_decode' not in vars(layer)
          and layer._decode.__func__ is module.GatedDeltaNet._decode, 'untouched exact GDN instance required')
    fused = _load_fused(module)
    return Installation(layer,make_decode(module,fused,enabled=True))
