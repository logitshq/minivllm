# Cold v3 sealing and route HTTP formal CPU evidence

Three complete guarded CPU attempts are retained: sealed cold v3 24/24 PASS; route formal v1 ran all 12 methods and reported 5 assertion failures (including subtest failures) plus 1 error; the complete v2 rerun passed 12/12. All attempts recorded zero skips, forbidden imports or real backend operations. The original failed result, tracebacks, commands, runner and source snapshots remain intact.

The only formal test repair canonicalized its temporary directory with Path.resolve(), covering macOS /var versus /private/var. The production formal source is byte-identical between attempts. Do not sum these reruns as 48 distinct tests.

The source-only seal records exactly four actual archived route-predecessor SHA constants and the regenerated lifecycle manifest. The archived before/after controller AST is identical after replacing those four constant values; the sealed sources match the 24-test snapshot. The seal records no tests or service operation. Its original evidence does not contain a separate actual invocation argv; seal.py is retained without inventing one. CPU invocation argv are preserved verbatim in each result and attempts.json.

evidence.tar.xz contains every file in all four original evidence trees, plus this archive script, command, summary and README. manifest.json maps source paths to member SHA256/size. archive-verification.json records every member reopened and matched to its original file. This is CPU wiring evidence only, not GPU, model numerical, HTTP runtime, performance or live restoration qualification.
