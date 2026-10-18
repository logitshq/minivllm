# oMLX formal CPU preparation

The original combined attempt ran 79 tests under the pinned oMLX vendor Python: 78 passed and one raised an error. The real prepared-CLI/inventory test correctly rejected that interpreter's `sys.prefix`, because the owner inventory belongs to the repository's outer `.venv`; the error was `outer runtime package missing: httpx`. This was an interpreter-selection mismatch, not a production-code repair.

The exact affected test then passed under the registered outer `.venv`. No production or test source changed. Both original runner files, commands, full stdout/stderr, result metadata and exact source snapshots are retained. This is 80 test executions across two attempts; the full 79-test suite was not rerun. The original FAIL remains FAIL.

The tests used CPU fixtures and guarded fake backends. Both receipts record zero real subprocesses, signals/network attempts and forbidden runtime modules. No model, GPU or real server ran. These results do not grant actual HTTP, numerical or performance qualification.

`evidence.tar.xz` contains every file from both original attempt directories. `manifest.json` maps every member to its source path, byte count and SHA-256. All archived members were read back and compared both by hash and byte for byte with their original files. The six sidecars are exact copies of the two result/stdout/stderr records; `archive-verification.json` records coverage and archive hashes.
