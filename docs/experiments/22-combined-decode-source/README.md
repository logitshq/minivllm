# V25 experiment snapshot

The 27 Python files under candidate/ are the exact combined, independently opt-in V23 conv+SiLU and V24 functional-decode source. Both features default OFF. This is an experimental snapshot, not replacement of the repository runtime or an adopted Pareto point.

source-manifest.json binds the executed source (SHA256 54f76862eb05d7ef8a28ad877afa43b4b08a1bb400c4347af18318c3767e06af). Paths inside receipts refer to the original local execution. Reproduction on another checkout requires adapting paths; it must not reuse old PASS receipts as fresh evidence.

evidence.tar.gz contains only the text files listed in evidence-index.json, copied verbatim. It includes numeric PASS, both the failed lifecycle harness attempt and corrected same-batch OFF/ON PASS, the frozen C1 harness, all six raw arms, resource records and external release proof. It excludes model weights, native binaries and raw tensor .bin files. The local original numeric receipt hashes remain inspectable; receipt replay is not independent recapture of those binary values.

The independent replay checked all36 HTTP requests/1152 output IDs, 102 per-arm raw resource samples and eight PID/six-port release evidence. Three paired medians were X+1.544470167% and Y+0.756317053%, with3/3 wins on both axes. This does not establish superiority over the official engines.
