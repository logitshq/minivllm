# Optional prospective V53 measurement-stability exploration
Evidence motivating this proposal is REVIEW.md:13–15. The physical cause remains unknown. No GPU execution is authorized or performed by this document.

Use exact V52 SG8 math,inputs,compiled A/B boundary,strict36 qualification,readonly checks,6AB/BA+4control pairs,repeats2048,2GiB cache and50/50/20/0. Change only per-block prime2048→8192. Keep identical prime for every arm/control; record four consecutive2048-call prime chunk envelopes outside timing to show whether drift persists. Do not select blocks/chunks or tune counts from results. Keep the measured2048-call block intact; source hashes/resource probes remain outside complete pairs.

Retain actual old controls:each ratio[.95,1.05],per-arm median[.98,1.02];gain>=5% and>=5/6wins. Keep STOP if controls fail; no rerun-until-pass,no threshold change,no old result pooling. One new fresh process/run with distinct source/output paths; expected~45–55s at recorded API wall,not guaranteed. It tests longer sustained work under the same probe-gap pattern; it cannot by itself identify clock/GC/GPU causes or prove E2E/model improvement.
