# Lithos D512 original-message retry

The first actual cycle is retained at commit0f64755. It rejected eight C1
requests because the harness sent already templated text as user.content.
This separate version sends fixture.messages and keeps the exact frozen
prompt_ids gate. It changes no model weights, pack, official inference code,
sampling setting, cache qualification, or resource/recovery policy.

Root CPU verification:63 protocol/trial/owner/qualification/resource checks and
28 controller/restoration checks PASS. Official CPU tokenizer reproduction
confirms every new C1/C8 request produces the original prompt IDs.

One observed C1 phase of8requests, then one observed C8 phase of16requests;
first failure stops, both direct servers must be reaped before restoring the
original oMLX. Preregistered cap512 with normal EOS; observers ON. This is
route/count/text/cache qualification only, not a formal Pareto measurement.

The verification inventory is byte-identical to v1. It includes all actual
runtime, model and pack dependencies; owner hashes them before/after execution.
The new restored predecessor is v1 PID25221, journal1798b9d3 and independently
passing restoration audit8365647c. Existing human stop/restore authorization
applies. Read-only preparation has succeeded; no stop occurred during prepare.

Run execute.py after committing this plan. It verifies receipt.json sources and
the exact PREPARED journal, executes the bounded owner and restores in finally.
