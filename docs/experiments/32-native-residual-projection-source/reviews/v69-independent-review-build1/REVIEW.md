# V69 independent source + saved numeric review
PASS_CPU_SOURCE_AND_SAVED_NUMERIC_REVIEW; no model/MLX import/GPU or original evidence modification.
All source-pins hashes rechecked; kernel_header.metal bytes equal frozen V66build2 original header17759af8…40c6b.
geometry.py:25–28 instantiates R=8,SG2,64threads; body:3 invokes original fp_qmv_fast_impl<T,16,4,false,R>.
Native header:280–307 keeps ascending K512 loop, qdot expression, FP4/scale decode, per-row FP32 accumulation, simd_sum and final BF16 cast.
R8 increases independent row accumulators/reuses16 loaded activation values across8 rows; per-row math source order is retained, occupancy/compiler effects need actual measurement.
Eligibility enforces complete N%(8*2)==0 and K%512==0 with original U32/U8 payload shapes; no partial-row OOB path for declared dimensions.
Source-eligible shape set is explicitly not GPU-qualified; actual scope is only L0 gate N17408/K5120,50,135,040B plus9 vectors.
Both A nativeQMM and B helper use fixed-shape mx.compile over same immutable W/scales, activation explicit; no math-policy override or factorfold/repack.
ensure_row_contiguous=True is wrapper behavior, not arbitrary input no-copy proof; prepare retains exact handles and observed active bytes unchanged.
Saved result3c8ef7e2…35dff is36strict finite BF16 checks; bytes.json has9 R4bridge +9compiledA/native +9compiledB/native +9B/A exact hash rows.
Actual trace counters remain A1/B1; readonly all11 input arrays before/after3 boundaries; qualification timing_allowedFalse and result timing_runFalse.
Release4fead37c…edac5 records84297/84298/84301 strict absence and toolrc0; saved resources retain AC/allowedthermal/pressure1 and stable swap/Swapouts.
Lithos gemv_T.metal:383–469 has RG-row activation reuse; gemv_tuning.py:16–19 permits rg8, default4. No evidence of actual M4 RG8 selection is asserted.
Future performance protocol is frozen prime2048/repeat8192,6AB/BA+4controls,gain≥5%+5wins with drift gates; none is satisfied or measured by this numeric receipt.
No universal exactness, fullmodel/HTTP/E2E/adoption or expected speed claim; no concrete source blocker found.
