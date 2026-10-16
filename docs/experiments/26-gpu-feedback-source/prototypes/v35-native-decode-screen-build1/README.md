# V35 B-only native decode exploratory screen

Root CLI: `.venv/bin/python screen.py --out NEW_ABSOLUTE_DIRECTORY --tokens 32`, exact frozen native1 DYLD injection + controls50/50/20/0. Accepted25/gates48ON/conv+whole-functionalOFF, one model load,2GiB allocator, one ordinary32 untimed initialization, one fixedcap256 graph capture. Native1/source25 are unchanged. Gate: actual V34three-request/93step strict qualification receipt SHA ff67fbc7669244ec061d0203540caafcc9d9bcffb1789b1f6e412431980730d5.

Schedule: six fresh independent19-token prefills, eachreset+all131seeduploads once, first3warm and final3measured. Each request then31native feedbacksteps observing only4Bprediction each. B consumes its own GPU argmax/position/lens/state feedback. Canonical32IDs asserted after timing. No A model calls in the timed decode.

t0 immediately beforeprefill; t1 afterfirsttoken/seedarray eval and native reset+upload; t2 after31native steps/4Bobservation. Raw decode31/(t2-t1), mean ns/step, and whole32/(t2-t0) are emitted separately. Whole scope includesfreshprefill,cachepreparation,reset andseedupload. Decode loop has no resources,logging,writes,statecopies,hashes,orcomparisons; onlynative calls,4Bobservation,listappend,anderrorchecks.

Aftert2: resourceguard then once-per-request copy/hash all132final staged outputs against saved lastbaseStep A_sha256 from the fixed qualification (reset-base final hashes also checked identical). These CPUcopies/hashes are explicitly outside all timing. Native counters enforce31steps/request,131uploads/reset,zero perstepCPUrestore and no comparator calls.

Resources originalraw precedederived/guard; beforeload/afterload-before-token/init/capture/requestpre-post/finally, knownAC/thermal0/pressure1/noforeign, strict no swap-used growth or Swapouts growth/reset acrosssamples. Root owns GPUexclusivity and external timeout/PIDcleanup, pausing other agents during actualtiming. B-only absolute exploratory timing only: no HTTP, Pareto, A/Bwin, or adoption claim.
