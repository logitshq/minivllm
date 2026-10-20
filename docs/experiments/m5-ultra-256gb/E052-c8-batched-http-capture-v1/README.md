# E052 v1: failed HTTP readiness identity, restored service

**Original experiment FAIL is preserved.** The parent, screen and trial all
failed; the owner/lease returned1. The capture stopped during readiness because
its expected identity contains `diagnostic_only: true` while the actual server
identity omits that key. Independent decoding of the saved readiness body confirms
that this is the only identity difference. No generation request was submitted.

There were seven readiness HTTP polls (six503, then one200), so "zero requests"
means zero inference/generation requests, not zero readiness GETs. The original
trace and route streams contain only one startup row each. No client directory,
group.json, joined-input or output tokens exist. No raw join was attempted, and
this run supplies no numerical or performance qualification. The expected-identity
repair belongs to a separate v2 namespace and cannot change this v1 outcome.

## Failure and recovery facts

The actual parent ran6.164382375s and exited1 (root tool85aab4/session95143 →937e57).
Original parent errors `screen/lease failed` and `closure: ` remain verbatim. The
trial's exact readiness ValueError, complete raw base64 body, startup identities,
server stdout/stderr and all source pins are retained. The saved server PID99766
was terminated and directly waited with returncode−15; strict absence and port
absence passed. The trial records cleanup_complete/server_reaped true. The lease
closed after its owner returned1 and resumed the same original oMLX PID52818 from
T to S with matching identity. All three saved post-health contracts pass.

`failure-resource-review.json` is **PASS_SAVED_FAILURE_RESTORE**, explicitly
separate from the experiment's FAIL. It uses unchanged frozen raw footprint,
command, metadata and health readers, while its control flow requires these exact
failure/exit1/readiness-error values. It never rewrites saved data into a PASS view.
Three physical and three metadata/checkpoint samples were checked. Maximum gap is
2.679061250s; sampled system maximum172,894,044,160B, child current14,189,048B and
child historical peak13,910,520B. Those last two are distinct recorded accounting
fields, not a derived estimate of the model's footprint. The brief spawn/reap
samples do not capture or bound the server's intervening loading peak.

Observed pressure1, thermal0, AC power, zero swap growth and Swapouts807478 satisfy
the registered64GiB child/240GiB system sampled thresholds. These are not allocation
caps. Resource/lease/health assertions refer to saved records; this review performs
no new live probe and does not independently reparse unavailable full historical
process inventories. The formal frontier remains unchanged.

## Archive and attempt preservation

The actual parent plan SHA is
`3e14ce911125eb46147943d5024510414271198984588dd91fea8319df8947e6`;
actual spec SHA is
`8f7d0095c98d70bb4d72609a1a91e48581da32fc0695e6363f0fadbafa0c281c`.
The original source preparation CPU run passed10/10 tests; that did not establish
runtime readiness compatibility. Its exact source snapshots/logs remain archived.
Original pending preparation fields are unchanged. `resource_audit.py` and
`archive-draft.py` are unexecuted success-path preparation sketches; the executed
failure-only reader is `failure_resource_audit.py` and archiver is `archive.py`.

An archive-source-generation heredoc failed before generating an executable
archiver (toolb59c2b, SyntaxError). Its actual tool error is retained in
`archive-source-preparation-failure.json`; the failed heredoc was not retained as
a separate script file. It caused no archive or device execution. The subsequent
actual resource review and first archive execution passed, with original commands,
stdout/stderr, elapsed times and returncodes saved. No failed attempt was replaced.

`evidence.tar.gz` has145 mapped members,3,252,822 original bytes compressed to
629,793B. Every member was independently reread and matched by size/SHA to both its
source and manifest; zero requested directories are missing and no payloads were
excluded. No model weights or generated tensors are present. The archive contains
original execution/screen/lease records, readiness and startup evidence, resource
raw observations, source pins, CPU evidence and independent failure review.

Archive SHA256:
`5df09b1c424ca03fb03eb0e69cb8bf63abc96fb1cda0f47c50d2f509a397a9fb`.
Manifest SHA256:
`c4ca253be774323486f3fe114f6b61650f5267557ccbb9321b2fedec348d4113`.
The outer `document-integrity.json` binds all side artifacts except itself.
This archival step performed no GPU, service, raw join or numerical replay work.
