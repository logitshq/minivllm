# E056: C1 shared MLP projection shadow canary

Actual run completed in 27.690361667 seconds total lifecycle, not an inference-latency metric. T4/T8 at layers 0,32,63 produced six real inputs, twelve native-vs-shared gate/up projection pairs, and thirty small raw tensor files. The independent audit reread all thirty files and passed all twelve pairs.

Shared shadow outputs never feed later layers. Native-returning traversal provenance compared full logits and 128 selected state arrays live, with successful full-state tensors not retained. Consequently this canary does not qualify an entire model using shared MLP projections, nor establish a speedup. E057 must do that full selected-prefix/continuation qualification before timing.

The separate saved-resource review passes nine physical/metadata samples; maximum observation gap 3.290103584s, max system bytes 192669237248, max child current bytes 19411186072, zero swap growth. The terminal observation/ACK/reap chain and original oMLX PID52818 restoration records pass. Sampling is not a hard allocation cap or proof about inter-sample peaks.

See `independent.json` and `resource-review.json`. Original complete raw tensors, producer receipts, owners and execution logs remain at `.artifacts/m5-ultra-256gb/c1-mlp-shared-canary-v1-*`; they have not yet been packaged into a complete distributable archive.
