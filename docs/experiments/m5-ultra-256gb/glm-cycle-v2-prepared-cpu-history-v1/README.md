# GLM v2 adapter CPU test history

Attempt1:15 PASS and1 fixture ERROR, retained unchanged. The negative fixture replaced an external plan with an incomplete dictionary, so validation raised KeyError earlier than the fixture expected. Only that fake plan was repaired to retain its original schema and add a changed value, allowing the intended SHA rejection to be exercised. No production code or validation gate changed. Attempt2:16/16 PASS,0 errors/failures/skips. Both actual guarded commands, stdout/stderr, source snapshots and before/after old-tree inventories are retained.

This archive was reopened member by member and checked against the original bytes/SHA, then original files and tree membership were rechecked. The archive step reran no tests. Neither attempt imported a device backend or performed real subprocess/socket/signal/model/service work. This evidence does not prove GLM load, model correctness, restoration, or speed.
