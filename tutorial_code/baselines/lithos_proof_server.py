"""Instrumentation-only launcher around the pinned official server.

Reads completed session programs; it never changes device/profile/kernel decisions.
"""
import json
from monolith.serve import Backend, main

original = Backend.complete


def observed_complete(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    session = self.session
    programs = []
    for key, engine in session.engines.items():
        program = engine.program
        matrix = []
        for op in program.ops:
            kernel = program.kernels[op.kernel]
            if kernel.function == "gemm_tile":
                matrix.append({"op": op.name, "accelerator": op.meta.get("accelerator"),
                    "COOP_A9": kernel.macros.get("COOP_A9"), "TM": kernel.macros.get("TM"),
                    "T_SRC": kernel.macros.get("T_SRC"), "T_LO": kernel.macros.get("T_LO"),
                    "T_HI": kernel.macros.get("T_HI"), "language_version": kernel.language_version})
        programs.append({"kind": str(key), "backend_id": program.backend_id,
            "config_digest": program.config_digest, "encoded_ops": len(program.ops),
            "gemm_tile_ops": len(matrix), "apple9_gemm_tile_ops": sum(x["COOP_A9"] == "1" for x in matrix),
            "tile_examples": matrix[:6]})
    evidence = {"session_accelerator_override": session.accelerator,
        "profile_accelerator": session.profile.accelerator, "profile": session.profile.name,
        "family": session.profile.family, "attention_override": session.attention,
        "profile_attention": session.profile.attention,
        "threadgroups_per_core": session.profile.threadgroups_per_core,
        "commute_norm": session.commute_norm, "programs": programs,
        "scope": "Read actual live Engines after complete() returned; their programs are compiled, encoded and generation completed. Predicated ops remain subject to recorded active T range; no isolated per-op GPU timestamp proof."}
    print("Lithos-Experiment-Proof: " + json.dumps(evidence), flush=True)
    return result


Backend.complete = observed_complete
main()
